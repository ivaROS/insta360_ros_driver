#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file detect_tag.py
@author Yanwei Du (yanwei.du@gatech.edu)
@date 11-26-2025
@version 1.0
@license Copyright (c) 2025
@desc None
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import apriltag
from math import sin, cos, atan2, asin, pi
from scipy.spatial.transform import Rotation

# ------------------------------
# CONFIGURATION
# ------------------------------

# Paths
IMAGE_DIR = "/mnt/IVALAB/rosbags/tsrb/revisits/2025-11-16-16-35-22/"  # folder containing .png images
TIMESTAMP_FILE = (
    "/mnt/IVALAB/rosbags/tsrb/revisits/2025-11-16-16-35-22/timestamps.txt"  # two columns: <timestamp> <image_name>
)
OUTPUT_POSE_FILE = "/mnt/IVALAB/rosbags/tsrb/revisits/2025-11-16-16-35-22/output_poses.txt"

# Equirectangular image parameters
# (If all images have same size, we can read from first image)
EQUIRECT_WIDTH = 5888
EQUIRECT_HEIGHT = 2944

# Virtual tile parameters
TILE_WIDTH = 1024
TILE_HEIGHT = 1024
TILE_FOV_DEG = 90.0  # horizontal FOV of virtual camera
TILE_OVERLAP_DEG = 20.0  # angular overlap between adjacent tiles (in yaw/pitch)

# Tag parameters
TAG_SIZE = 0.19  # meters (edge length)


# Camera intrinsics for each tile (assume symmetric fx, fy)
def get_tile_intrinsics():
    fx = (TILE_WIDTH / 2) / np.tan(np.deg2rad(TILE_FOV_DEG) / 2)
    fy = fx
    cx = TILE_WIDTH / 2
    cy = TILE_HEIGHT / 2
    return fx, fy, cx, cy


# Pose fusion
POSE_FUSION_MODE = "average"  # "average" or "best"
REPROJ_ERROR_THRESH = 5.0  # pixels; filter out bad tile poses

# Visualization
ENABLE_VIS = True  # set True to show tiles with detections


def average_poses_SE3(poses):
    """
    poses: list of (R, t) where R is 3x3, t is 3-vector.
    Return averaged (R, t) using naive quaternion averaging + mean translation.
    """
    if len(poses) == 1:
        return poses[0]

    # Average translation
    ts = np.stack([t for (_, t) in poses], axis=0)
    t_mean = np.mean(ts, axis=0)

    # Average rotation using quaternions
    quats = []
    for R, _ in poses:
        q = Rotation.from_matrix(R).as_quat()
        quats.append(q)
    quats = np.stack(quats, axis=0)

    # Ensure they lie on same hemisphere
    q0 = quats[0]
    for i in range(1, quats.shape[0]):
        if np.dot(q0, quats[i]) < 0:
            quats[i] = -quats[i]

    q_mean = np.mean(quats, axis=0)
    q_mean /= np.linalg.norm(q_mean)

    # Convert back to R
    R_mean = Rotation.from_quat(q_mean).as_matrix()
    return R_mean, t_mean


# ------------------------------
# SPHERE / EQUIRECT / TILE GEOMETRY
# ------------------------------


def equirect_shape_from_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image {path}")
    return img.shape[1], img.shape[0]  # width, height


def get_tile_extrinsics(yaw, pitch):
    R_yaw = Rotation.from_euler("y", yaw).as_matrix()
    R_pitch = Rotation.from_euler("x", pitch).as_matrix()
    R_cam = Rotation.from_euler("x", np.pi).as_matrix()  # camera looks along -Z
    R_wc = R_yaw @ R_pitch @ R_cam
    return R_wc


def dir_from_spherical(lon, lat):
    """
    lon, lat in radians.
    lon = yaw in [-pi, pi] (0 along +Z, positive to +X).
    lat = in [-pi/2, pi/2], positive upward.
    Returns 3D unit direction in omni frame C with:
      x right, y up, z forward.
    """
    x = cos(lat) * sin(lon)
    y = sin(lat)
    z = cos(lat) * cos(lon)
    return np.array([x, y, z])


def spherical_from_dir(d):
    """
    Inverse of dir_from_spherical. d: 3-vector (unit).
    Returns (lon, lat).
    """
    x, y, z = d
    lon = atan2(x, z)
    lat = asin(y)
    return lon, lat


def generate_tile_orientations(tile_fov_deg, overlap_deg):
    """
    Generate yaw/pitch centers (in radians) of tiles that cover the sphere.
    We cover yaw in [-pi, pi), pitch in [-pi/2, pi/2].
    """
    step = np.deg2rad(tile_fov_deg - overlap_deg)
    if step <= 0:
        raise ValueError("Overlap too large, step becomes non-positive.")

    yaws = np.arange(-pi, pi, step)
    pitches = np.arange(pi / 2 - step, pi / 2 + step, step)
    # If you want full coverage including poles, you may need special pole tiles.
    tile_orients = []
    for pitch in pitches:
        for yaw in yaws:
            tile_orients.append((yaw, pitch))
    return tile_orients


def tile_rotation_C_from_yaw_pitch(yaw, pitch):
    """
    Build rotation R_C_Ti: from tile camera frame Ti to omni camera frame C.
    - Tile camera: +Z backward, +X right, +Y down (image coords y down).
    - Omni frame:  +Z forward, +X right, +Y up.
    We define the forward direction of tile as dir_from_spherical(yaw, pitch).
    """
    # Forward (z_cam) in omni frame
    z_cam = dir_from_spherical(yaw, pitch)
    z_cam /= np.linalg.norm(z_cam)

    # Define an up reference in omni frame (world up)
    up_world = np.array([0.0, 1.0, 0.0])

    # x_cam = up_world x z_cam (right)
    x_cam = np.cross(up_world, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        # z_cam is near up_world; pick a different up axis
        up_world = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(up_world, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    # y_cam (up in omni) = z_cam x x_cam
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    # Tile uses +Y down for image coordinates, so flip y_cam
    y_cam = -y_cam

    # Columns are the basis vectors of tile frame expressed in omni frame
    R_C_Ti = np.stack([x_cam, y_cam, z_cam], axis=1)  # 3x3
    return R_C_Ti


def render_perspective_tile(eqr_img, yaw, pitch, tile_width, tile_height, tile_fov_deg):
    """
    Render a perspective tile from the equirectangular image eqr_img
    at the given yaw/pitch (tile center).
    Returns tile_img (uint8) and R_C_Ti (rotation).
    """
    H_eqr, W_eqr = eqr_img.shape[:2]
    fx, fy, cx, cy = get_tile_intrinsics()

    # R_C_Ti = tile_rotation_C_from_yaw_pitch(yaw, pitch)
    R_C_Ti = get_tile_extrinsics(yaw, pitch)

    # Create grid of pixel coordinates in tile
    u = np.arange(tile_width)
    v = np.arange(tile_height)
    uu, vv = np.meshgrid(u, v)

    # Convert to normalized camera coordinates in tile frame
    x_t = (uu - cx) / fx
    y_t = (vv - cy) / fy
    z_t = np.ones_like(x_t)

    # Directions in tile frame
    dir_T = np.stack([x_t, y_t, z_t], axis=-1)
    norm = np.linalg.norm(dir_T, axis=-1, keepdims=True)
    dir_T /= norm

    print(dir_T.shape)

    # Directions in omni frame: dir_C = R_C_Ti * dir_T
    dir_C = dir_T @ R_C_Ti.T  # (H,W,3)

    # Convert directions to equirectangular spherical coords
    x_c = dir_C[..., 0]
    y_c = dir_C[..., 1]
    z_c = dir_C[..., 2]

    lon = np.arctan2(x_c, z_c)  # [-pi, pi]
    lat = np.arcsin(np.clip(y_c, -1.0, 1.0))  # [-pi/2, pi/2]

    # Map lon, lat to pixel coords in equirect
    # Insta360 unwrap image CW from [0, -2pi]
    u_eqr = (2 * pi - lon) / (2 * pi) * W_eqr
    v_eqr = (pi / 2 - lat) / pi * H_eqr

    # Remap using cv2.remap (need float32)
    map_x = u_eqr.astype(np.float32)
    map_y = v_eqr.astype(np.float32)
    tile_img = cv2.remap(eqr_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    return tile_img, R_C_Ti


# ------------------------------
# APRILTAG POSE FROM TILE
# ------------------------------


def estimate_camera_pose_from_tile(tile_img, R_C_Ti, detector):
    """
    tile_img: grayscale tile
    R_C_Ti:  3x3 rotation from tile frame Ti to omni frame C
    Returns:
      - list of (R_C, t_C) poses for the camera in omni frame
      - list of reprojection errors for each pose
    If you only want one pose per tile (e.g. multiple tags fused), you can adapt.
    """
    fx, fy, cx, cy = get_tile_intrinsics()
    camera_params = (fx, fy, cx, cy)

    detections = detector.detect(tile_img)
    poses_C = []
    errors = []

    # Example: treat each detection independently, get one pose per detection
    for det in detections:
        # python-apriltag pose API varies; in many forks it's:
        # pose, e0, e1 = detector.detection_pose(det, camera_params, TAG_SIZE)
        try:
            pose, e0, e1 = detector.detection_pose(det, camera_params, TAG_SIZE)
        except TypeError:
            # If your apriltag version uses different signature, adapt here
            continue

        R_Ti_cam = pose[0:3, 0:3]
        t_Ti_cam = pose[0:3, 3]

        # NOTE: you must check what 'pose' means in your apriltag version:
        # - If it's camera w.r.t tag or tag w.r.t camera.
        # Assume here pose is camera in tile frame: T_Ti_cam
        # If that's wrong, you'll have to invert accordingly.

        # Camera pose in omni frame:
        # R_C_cam = R_C_Ti * R_Ti_cam
        R_C_cam = R_C_Ti @ R_Ti_cam
        # Origin of camera in omni: t_C_cam = R_C_Ti * t_Ti_cam (no translation between C and Ti)
        t_C_cam = R_C_Ti @ t_Ti_cam

        # Use (e0+e1)/2 as a crude reprojection error
        reproj_err = 0.5 * (e0 + e1)
        poses_C.append((R_C_cam, t_C_cam))
        errors.append(reproj_err)

    return poses_C, errors


# ------------------------------
# VISUALIZATION
# ------------------------------


def visualize_tile_with_detections(tile_img, detections):
    plt.figure()
    plt.imshow(tile_img, cmap="gray")
    for det in detections:
        # det.corners is 4x2 array
        corners = det.corners
        xs = corners[:, 0]
        ys = corners[:, 1]
        plt.plot(xs[[0, 1, 2, 3, 0]], ys[[0, 1, 2, 3, 0]], "-")
        cx, cy = det.center
        plt.text(cx, cy, str(det.tag_id), color="red")
    plt.title("Tile with AprilTag detections")
    # plt.gca().invert_yaxis()
    plt.show()


# ------------------------------
# MAIN OFFLINE PIPELINE
# ------------------------------


def load_timestamps(timestamp_file, image_dir):
    """
    Expect file with lines: <timestamp> <image_name>
    Returns list of (timestamp, full_image_path).
    """
    entries = []
    with open(timestamp_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts_str, img_name = line.split()
            ts = float(ts_str)
            img_path = os.path.join(image_dir, img_name)
            entries.append((ts, img_path))
    return entries


def main():
    global EQUIRECT_WIDTH, EQUIRECT_HEIGHT

    # Load timestamp + image list
    entries = load_timestamps(TIMESTAMP_FILE, IMAGE_DIR)
    if not entries:
        print("No entries found.")
        return

    # Get equirect size from first image
    _, first_img_path = entries[0]
    EQUIRECT_WIDTH, EQUIRECT_HEIGHT = equirect_shape_from_image(first_img_path)

    # Setup AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    # Pre-generate tile orientations
    tile_orients = generate_tile_orientations(TILE_FOV_DEG, TILE_OVERLAP_DEG)
    print(f"Using {len(tile_orients)} tiles per image.")

    # Prepare output file
    out_f = open(OUTPUT_POSE_FILE, "w")
    out_f.write("# timestamp tx ty tz qx qy qz qw\n")

    for idx, (timestamp, img_path) in enumerate(entries):
        print(f"[{idx+1}/{len(entries)}] Processing {img_path}")
        eqr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if eqr_img is None:
            print(f"Failed to read {img_path}, skipping.")
            continue

        # For this image: collect all camera poses from all tiles
        image_poses = []
        image_errors = []

        for yaw, pitch in tile_orients:
            print(yaw, pitch)
            tile_img, R_C_Ti = render_perspective_tile(eqr_img, yaw, pitch, TILE_WIDTH, TILE_HEIGHT, TILE_FOV_DEG)

            # cv2.imshow("Tile", tile_img)
            # cv2.waitKey()

            poses_C, errors = estimate_camera_pose_from_tile(tile_img, R_C_Ti, detector)
            for (R, t), e in zip(poses_C, errors):
                if e < REPROJ_ERROR_THRESH:
                    image_poses.append((R, t))
                    image_errors.append(e)

            if ENABLE_VIS and poses_C:
                detections = detector.detect(tile_img)
                visualize_tile_with_detections(tile_img, detections)

        if not image_poses:
            print("  No good poses found for this image.")
            continue

        # Fuse poses for this image
        if POSE_FUSION_MODE == "best":
            best_idx = int(np.argmin(image_errors))
            R_best, t_best = image_poses[best_idx]
            R_img, t_img = R_best, t_best
        else:  # "average"
            R_img, t_img = average_poses_SE3(image_poses)

        # Convert to quaternion (wxyz)
        # q = rotmat_to_quat_wxyz(R_img)
        q = Rotation.from_matrix(R_img).as_quat()

        # Save as: timestamp tx ty tz qx qy qz qw
        out_line = (
            f"{timestamp:.9f} {t_img[0]:.6f} {t_img[1]:.6f} {t_img[2]:.6f} "
            f"{q[1]:.6f} {q[2]:.6f} {q[3]:.6f} {q[0]:.6f}\n"
        )
        out_f.write(out_line)

    out_f.close()
    print(f"Saved poses to {OUTPUT_POSE_FILE}")


if __name__ == "__main__":
    main()
