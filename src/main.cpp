#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <camera/photography_settings.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>

#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    ros::NodeHandle node_;
    ros::Publisher  compressed_pub_;
    ros::Publisher  imu_pub_;

public:
    TestStreamDelegate(const ros::NodeHandle& node) : node_(node) {
        // Publisher for the compressed H.264 video stream
        compressed_pub_ = node_.advertise<sensor_msgs::CompressedImage>(
            "/insta/image/compressed", 10);

        // Publisher for IMU data (remains the same)
        imu_pub_ =
            node_.advertise<sensor_msgs::Imu>("/insta/imu/data_raw", 100);
        ROS_INFO("Publisher for compressed images and IMU created.");
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size,
                     int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp,
                     uint8_t streamType, int stream_index) override {
        // We only care about the main video stream (index 0)
        if (stream_index == 0 && size > 0) {
            sensor_msgs::CompressedImage msg;

            // Set the header
            msg.header.stamp    = ros::Time::now();
            msg.header.frame_id = "camera_frame";

            // Set the format to H.264
            // The subscriber will need to know this to select the correct
            // decoder.
            msg.format = "h264";

            // Copy the compressed video data directly into the message
            msg.data.assign(data, data + size);

            compressed_pub_.publish(msg);
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        for (const auto& gyro : data) {
            sensor_msgs::Imu msg;
            msg.header.stamp       = ros::Time::now();
            msg.header.frame_id    = "imu_frame";
            msg.angular_velocity.x = gyro.gx;
            msg.angular_velocity.y = gyro.gy;
            msg.angular_velocity.z = gyro.gz;

            msg.linear_acceleration.x = gyro.ax * 9.80665;
            msg.linear_acceleration.y = gyro.ay * 9.80665;
            msg.linear_acceleration.z = gyro.az * 9.80665;

            msg.orientation.x = 0.0;
            msg.orientation.y = 0.0;
            msg.orientation.z = 0.0;
            msg.orientation.w = 1.0;  // Neutral orientation
            msg.orientation_covariance[0] =
                -1.0;  // No orientation data available

            for (int i = 0; i < 9; i++) {
                msg.angular_velocity_covariance[i]    = 0;
                msg.linear_acceleration_covariance[i] = 0;
            }
            // imu_pub_.publish(msg);
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    ros::NodeHandle                     node_;

public:
    CameraWrapper(const ros::NodeHandle& node) : node_(node) {}

    ~CameraWrapper() {
        if (cam) {
            cam->Close();
        }
    }

    int run_camera() {
        ins_camera::DeviceDiscovery discovery;
        auto                        list = discovery.GetAvailableDevices();
        if (list.empty()) {
            ROS_ERROR("No available camera devices found.");
            return -1;
        }

        cam = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam->Open()) {
            ROS_ERROR("Failed to open camera.");
            return -1;
        }
        ROS_INFO("Camera opened successfully.");
        discovery.FreeDeviceDescriptors(list);

        std::shared_ptr<ins_camera::StreamDelegate> delegate =
            std::make_shared<TestStreamDelegate>(node_);
        cam->SetStreamDelegate(delegate);

        auto start = time(NULL);
        cam->SyncLocalTimeToCamera(start);
        ins_camera::LiveStreamParam param;
        param.video_resolution = ins_camera::VideoResolution::
            RES_3840_1920P30;  // Change this line to edit the resolution
        // Possible resolutions (results may vary per model) are:
        // RES_3840_1920P30
        // RES_2560_1280P30
        // RES_1152_1152P30 (this will give 2304 x 1152 at 30 FPS)
        // RES_1920_960P30
        param.lrv_video_resulution =
            ins_camera::VideoResolution::RES_1440_720P30;
        param.video_bitrate = 1024 * 1024 / 2;
        param.enable_audio  = false;
        param.using_lrv     = false;

        if (!cam->StartLiveStreaming(param)) {
            ROS_ERROR("Failed to start live streaming.");
            return -1;
        }

        ROS_INFO("Live streaming started.");
        return 0;
    }
};

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "insta_publisher");
    ros::NodeHandle nh("~");
    CameraWrapper   camera(nh);
    if (camera.run_camera() != 0) {
        ros::shutdown();
        return -1;
    }
    ros::spin();
    ros::shutdown();
    return 0;
}