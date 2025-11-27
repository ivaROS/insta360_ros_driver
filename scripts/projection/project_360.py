import cv2
import numpy as np

from pathlib import Path
import glob
import apriltag


def equirect_perspective_projector(source_scale, yaw, pitch, intrinsics, image_size):
    """
    Construct a projection lookup table for projecting
        equirectangular images to perspective views.

    @param yaw: yaw angle of view to generate
    @param pitch: pitch angle of view to generate
    @param intrinsics: 2x3 intrinsics matrix (ndarray)
    @param image_size: 2-tuple (width, height)

    @return Image projector function that takes in the desired image,
            and gives a perspective image.
    """
    px_xs, px_ys = np.meshgrid(range(image_size[0]), range(image_size[1]))

    # XYZ are in camera coords: X going to the right, Y going down,
    #     Z going into the picture.
    xs = (px_xs - intrinsics[0, 2]) / intrinsics[0, 0]
    ys = (px_ys - intrinsics[1, 2]) / intrinsics[1, 1]
    zs = np.ones_like(xs)
    pitch_mat = np.array([[1, 0, 0], [0, np.cos(pitch), np.sin(pitch)], [0, -np.sin(pitch), np.cos(pitch)]])
    yaw_mat = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
    # transpose -> row, col, color
    xyz = np.stack([xs, ys, zs]).transpose((1, 2, 0))
    xyz = xyz @ (pitch_mat.T @ yaw_mat.T)
    horiz_dist = np.linalg.norm(xyz[:, :, (0, 2)], axis=2)

    # 0 to 2pi
    unproj_yaw = np.arctan2(xyz[:, :, 0], xyz[:, :, 2]) + np.pi
    # 0 to pi
    unproj_pitch = np.arctan2(xyz[:, :, 1], horiz_dist) + (np.pi / 2)

    R = source_scale / (np.pi)
    map_x = unproj_yaw * R
    map_y = unproj_pitch * R

    # NOTE: convertMaps requires float32 array arguments
    imap_x, imap_y = cv2.convertMaps(
        np.array(map_x, dtype=np.float32), np.array(map_y, dtype=np.float32), cv2.CV_16SC2, cv2.CV_16UC1
    )

    def project(source):
        if len(source.shape) == 2:
            dest_img = np.zeros(image_size[::-1], dtype=source.dtype)
        else:
            dest_img = np.zeros([*image_size[::-1], source.shape[2]], dtype=source.dtype)
        cv2.remap(source, imap_x, imap_y, cv2.INTER_LINEAR, dest_img, cv2.BORDER_WRAP)
        return dest_img

    return project


def equirect_to_perspective(source, yaw, pitch, intrinsics, image_size):
    """
    Project equirectangular image to perspective.

    Yaw angle is measured, zero from the center of the image,
        increasing to the left (CCW from robot-standard Z axis)
    Pitch angle is measured zero in the middle of the image,
        positive downwards, negative upwards. (CCW from robot-standard Y axis)
    Intrinsics is a 2x3 camera matrix [(fx, 0, cx), (0, fy, cy)]
    Image size is (width, height).

    Maybe instead of yaw, pitch, we should just take a user input transform matrix?

    @param yaw: yaw angle of view to generate
    @param pitch: pitch angle of view to generate
    @param intrinsics: 2x3 intrinsics matrix (ndarray)
    @param image_size: 2-tuple (width, height)

    @return projected image
    """
    source_height, source_width = source.shape[0], source.shape[1]
    assert source_height == source_width // 2
    projector = equirect_perspective_projector(source_height, yaw, pitch, intrinsics, image_size)
    return projector(source)


if __name__ == "__main__":
    import os
    import yaml

    # data_file = os.path.expanduser("~/datasets/360_img/IMG_20250711_221307_00_026.jpg")
    # prefix = "/home/yanwei/Documents/cid/360images"
    prefix = "/mnt/IVALAB/rosbags/tsrb/revisits/2025-11-16-16-35-22/"
    filenames = sorted(glob.glob(f"{prefix}/*.jpg"))
    # data_file = os.path.expanduser("~/datasets/360_img/IMG_20250711_221307_00_026.jpg")

    detector = apriltag.Detector()
    for data_idx, data_file in enumerate(filenames):
        intrinsics = np.array(
            [
                [512, 0, 512],
                [0, 512, 512],
            ]
        )

        equirect_img = cv2.imread(data_file)
        print(data_file)

        import time

        N = 1
        yaw = np.random.random() * 2 * np.pi
        pitch = -np.pi / 2.0  # 0#np.random.random() * np.pi - (np.pi/2)
        print(yaw, pitch)
        im_sz = (1024, 1024)
        t1 = time.monotonic()
        projector1 = equirect_perspective_projector(equirect_img.shape[0], yaw, pitch, intrinsics, im_sz)
        projector2 = equirect_perspective_projector(equirect_img.shape[0], yaw + np.pi / 2, pitch, intrinsics, im_sz)
        projector3 = equirect_perspective_projector(equirect_img.shape[0], yaw + np.pi, pitch, intrinsics, im_sz)
        projector4 = equirect_perspective_projector(equirect_img.shape[0], yaw - np.pi / 2, pitch, intrinsics, im_sz)
        t2 = time.monotonic()
        for i in range(N):
            render1 = projector1(equirect_img)
            render2 = projector2(equirect_img)
            render3 = projector3(equirect_img)
            render4 = projector4(equirect_img)
        t3 = time.monotonic()

        print(f"Construct time: {t2 - t1}")
        print(f"Compiled time: {t3 - t2} per {N} frames")

        import matplotlib.pyplot as plt

        render = render1
        result1, render1 = detector.detect(cv2.cvtColor(render1, cv2.COLOR_BGR2GRAY), return_image=True)
        result2, render2 = detector.detect(cv2.cvtColor(render2, cv2.COLOR_BGR2GRAY), return_image=True)
        result3, render3 = detector.detect(cv2.cvtColor(render3, cv2.COLOR_BGR2GRAY), return_image=True)
        result4, render4 = detector.detect(cv2.cvtColor(render4, cv2.COLOR_BGR2GRAY), return_image=True)

        fig, ((a1, a2, a3, a4, a5)) = plt.subplots(1, 5)
        a1.axis("off")
        a2.axis("off")
        a3.axis("off")
        a4.axis("off")
        a1.imshow(cv2.cvtColor(render4, cv2.COLOR_BGR2RGB))
        a2.imshow(cv2.cvtColor(render3, cv2.COLOR_BGR2RGB))
        a3.imshow(cv2.cvtColor(render2, cv2.COLOR_BGR2RGB))
        a4.imshow(cv2.cvtColor(render1, cv2.COLOR_BGR2RGB))
        a5.imshow(cv2.cvtColor(render, cv2.COLOR_BGR2RGB))

        # plt.figure(2)
        # plt.imshow(cv2.cvtColor(render, cv2.COLOR_BGR2RGB))
        # plt.tight_layout()
        # fig.savefig(f"{prefix}/tags/{data_idx}", dpi=300)
        plt.show()
