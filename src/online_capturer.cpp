/**
 * @file online_capturer.cpp
 * @author duyanwei (duyanwei0702@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-10-01
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <camera/photography_settings.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <std_srvs/Trigger.h>
#include <unistd.h>

#include <atomic>
#include <boost/filesystem.hpp>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>
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
        //        ROS_INFO("Publisher for compressed images and IMU created.");
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size,
                     int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp,
                     uint8_t streamType, int stream_index) override {
        return;
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

            // compressed_pub_.publish(msg);
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        return;
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
public:
    /**
     * @brief Construct a new Camera Wrapper object
     *
     * @param node
     */
    CameraWrapper(const ros::NodeHandle& node) : node_(node), cam_(nullptr) {}

    /**
     * @brief Destroy the Camera Wrapper object
     *
     */
    ~CameraWrapper() {
        // Save Stamps
        {
            std::string   filename{output_dir_ + "/timestamps.txt"};
            std::ofstream myfile(filename);
            if (myfile.is_open()) {
                for (const auto& s : stamps_) {
                    myfile << s << "\n";
                }
                myfile.close();
            }
            std::cout << "Images with timestamps have been saved to: " << output_dir_ << std::endl;
        }
        if (cam_) {
            cam_->Close();
            std::cout << "Camera has been closed." << std::endl;
        }
    }

    int InitCam() {
        ROS_INFO("Start to open Camera ...");
        ins_camera::DeviceDiscovery discovery;
        auto                        list = discovery.GetAvailableDevices();
        if (list.empty()) {
            ROS_ERROR("No available camera devices found.");
            return -1;
        }

        cam_ = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam_->Open()) {
            ROS_ERROR("Failed to open camera.");
            return -1;
        }
        ROS_INFO("Camera opened successfully.");
        discovery.FreeDeviceDescriptors(list);

        std::shared_ptr<ins_camera::StreamDelegate> delegate =
            std::make_shared<TestStreamDelegate>(node_);
        cam_->SetStreamDelegate(delegate);

        auto start = time(NULL);
        cam_->SyncLocalTimeToCamera(start);
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

        if (!cam_->StartLiveStreaming(param)) {
            ROS_ERROR("Failed to start live streaming.");
            return -1;
        }

        // Enable online stitching.
        bool enable_stitching = true;
        if (cam_->EnableInCameraStitching(enable_stitching)) {
            ROS_INFO("In-camera stitching function activated successfully!");
        } else {
            ROS_WARN("In-camera stitching is NOT activated!");
        }

        ROS_INFO("Live streaming started.");
        return 0;
    }

    void InitRos() {
        enable_service_      = node_.param<bool>("enable_service", false);
        enable_timer_        = node_.param<bool>("enable_timer", false);
        enable_pose_checker_ = node_.param<bool>("enable_pose_checker", false);

        time_thresh_  = node_.param<float>("time_thresh", 5);     // seconds
        trans_thresh_ = node_.param<float>("trans_thresh", 2.0);  // meters

        if (enable_service_) {
            enable_timer_        = false;
            enable_pose_checker_ = false;
            service_             = node_.advertiseService(
                "/capture", &CameraWrapper::captureRequest, this);
            ROS_INFO("Service mode enabled!!!");
        } else if (enable_pose_checker_) {
            odom_sub_ =
                node_.subscribe("/odom", 10, &CameraWrapper::odomCb, this);
            enable_timer_ = false;
            ROS_INFO("PoseChecker mode enabled!!!");
        }

        if (enable_timer_) {
            time_thresh_ = time_thresh_ < 0.0 ? 5.0 : time_thresh_;
            timer_       = node_.createTimer(ros::Duration(time_thresh_),
                                             &CameraWrapper::timerCallback, this);
            ROS_INFO("Timer mode enabled!!!");
        }

        // ROS Parameters.
        output_dir_ =
            node_.param<std::string>("output_dir", "/tmp/insta/");  // meters
        if (!endsWithSlash(output_dir_)) {
            output_dir_.append("/");
        }

        output_dir_ += getCurrentTimeAsDirectoryString();
        output_dir_.append("/");
        boost::filesystem::create_directories(output_dir_);
    }

    /**
     * @brief
     *
     * @param event
     */
    void timerCallback(const ros::TimerEvent& event) { capture(); }

    void odomCb(const nav_msgs::OdometryConstPtr& msg) {
        float cur_tx = msg->pose.pose.position.x;
        float cur_ty = msg->pose.pose.position.y;

        if (last_pose_.isValid()) {
            float dist2 = (cur_tx - last_pose_.tx) * (cur_tx - last_pose_.tx) +
                          (cur_ty - last_pose_.ty) * (cur_ty - last_pose_.ty);
            if (dist2 > trans_thresh_ * trans_thresh_) {
                capture();
                last_pose_.tx = cur_tx;
                last_pose_.ty = cur_ty;
                last_pose_.timestamp = msg->header.stamp.toSec();
            }
        } else {

        // thread safe.
        last_pose_.timestamp = msg->header.stamp.toSec();
        last_pose_.tx        = msg->pose.pose.position.x;
        last_pose_.ty        = msg->pose.pose.position.y;
        }
    }

    bool captureRequest(std_srvs::Trigger::Request&  req,
                        std_srvs::Trigger::Response& res) {
        ROS_INFO("Capturing Service is Triggered!");
        res.success = capture();
        res.message = res.success ? "Image Captured" : "Fail to capture!";
        return true;
    }

    /**
     * @brief
     *
     */
    bool capture() {
        if (!cam_) {
            ROS_ERROR("Camera is not initialized!");
            return false;
        }
        bool ret =
            cam_->SetPhotoSubMode(ins_camera::SubPhotoMode::PHOTO_SINGLE);
        if (!ret) {
            ROS_ERROR("change sub mode failed!");
            return false;
        }

        ROS_INFO("Taking a picture ...");
        double     timestamp = ros::Time::now().toSec();
        const auto url       = cam_->TakePhoto();
        if (!url.IsSingleOrigin() || url.Empty()) {
            ROS_ERROR("failed to take picture");
            return false;
        }

        const std::string download_url = url.GetSingleOrigin();
        const std::string file_name    = this->getFileName(download_url);

        std::string save_path = output_dir_ + file_name;
        ret = cam_->DownloadCameraFile(download_url, save_path);
        if (ret) {
            ROS_INFO_STREAM("Download " << download_url << " succeed!!!");
            ROS_INFO_STREAM("Image saved to: " << save_path);
            stamps_.emplace_back(timestamp, file_name);
        } else {
            ROS_ERROR_STREAM("Download " << download_url << " failed!!!");
        }
        return true;
    }

    void run() { ros::spin(); }

private:
    std::string getCurrentTimeAsDirectoryString() {
        ros::Time   now = ros::Time::now();
        std::time_t t   = (time_t)now.sec;  // Convert ROS time to std::time_t

        std::stringstream ss;
        // Format as YYYY-MM-DD-HH-MM-SS
        ss << std::put_time(std::localtime(&t), "%Y-%m-%d-%H-%M-%S");
        return ss.str();
    }

    std::string getFileName(const std::string& path) {
        std::smatch sm;
        std::string dir, name;
        std::regex_match(path, sm, std::regex("(.+?)([^\\/\\\\]+$)"));
        if (sm.size() <= 2) {
            return path;
        }
        return sm[2].str();
    }

    bool endsWithSlash(const std::string& path) {
        if (path.empty()) {
            return false;
        }
        return path.back() == '/' || path.back() == '\\';
    }

    struct StampedImage {
        double      timestamp;
        std::string filename;

        StampedImage() = default;
        StampedImage(double _timestamp, const std::string& _filename)
            : timestamp(_timestamp), filename(_filename) {}

        friend std::ostream& operator<<(std::ostream&       os,
                                        const StampedImage& s) {
            os << std::fixed << std::setprecision(6) << s.timestamp << " "
               << s.filename;
            return os;
        }
    };

    struct StampedPose {
        double timestamp = -1.0;
        double tx = 0.0, ty = 0.0, theta = 0.0;
        bool   isValid() const { return timestamp >= 0.0; }
    };

    ros::NodeHandle                     node_;
    std::shared_ptr<ins_camera::Camera> cam_;
    std::vector<StampedImage>           stamps_;
    ros::ServiceServer                  service_;
    ros::Subscriber                     odom_sub_;
    ros::Timer                          timer_;

    std::string output_dir_{"/tmp/insta/"};
    bool        enable_service_{false};
    bool        enable_pose_checker_{false};
    bool        enable_timer_{false};
    std::mutex  pose_mutex_;
    StampedPose last_pose_;
    float       time_thresh_{-1.0};
    float       trans_thresh_{-1.0};
    float       rot_thresh_{-1.0};
};

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "insta_capturer");
    ros::NodeHandle nh("~");
    CameraWrapper   camera(nh);
    if (camera.InitCam() != 0) {
        ros::shutdown();
        return -1;
    }
    camera.InitRos();
    camera.run();
    return 0;
}
