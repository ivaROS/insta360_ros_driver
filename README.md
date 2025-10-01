# insta360_ros_driver

A ROS driver for the Insta360 cameras. This driver is tested on Ubuntu 20.04 with ROS Noetic.

## Installation
To use this driver, you need to first have Insta360 SDK. Please apply for the SDK from the [Insta360 website](https://www.insta360.com/sdk/home). 

```
cd ~/catkin_ws/src
git clone git@github.com:ivaROS/insta360_ros_driver.git
cd ..
```
Then, the Insta360 libraries need to be installed as follows:
- add the <code>camera</code> and <code>stream</code> header files inside the <code>include</code> directory
- add the <code>libCameraSDK.so</code> library under the <code>lib</code> directory.

Afterwards, install the other required dependencies and build
```
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

Before continuing, **make sure the camera is set to dual-lens mode**

The Insta360 requires **sudo privilege** to be accessed via USB. To compensate for this, a udev configuration can be automatically created that will only request for sudo once. The camera can thus be setup initially via:
```
./setup.sh
```
This creates a symlink  based on the vendor ID of Insta360 cameras. The symlink, in this case <code>/dev/insta</code> is used to grant permissions to the usb port used by the camera.
![setup](docs/setup.png)

## Usage
Launch the capturer:
```bash
roslaunch insta360_ros_driver online_capture.launch
```

The capture script supports three trigger modes, checked in the following priority:

1. **Service** – capture is triggered via a service call from other nodes.

2. **Pose checker** – monitors the `/odom` topic and triggers capture when translation exceeds a threshold (`trans_thresh`).

3. **Timer** – captures images at a fixed interval set by a timer (`time_thresh`).
Parameters

Parameters:
- `output_dir`: directory where the images are saved.
- `time_thresh` (seconds): interval for timer-based capture.

- `trans_thresh` (meters): translation threshold for pose-based capture.