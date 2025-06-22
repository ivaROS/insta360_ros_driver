#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
// #include <sensor_msgs/ImageEncodings.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

static enum AVPixelFormat get_hw_format(AVCodecContext*           ctx,
                                        const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

class H264DecoderNode {
private:
    AVCodec*              codec_      = nullptr;
    AVCodecContext*       codec_ctx_  = nullptr;
    AVCodecParserContext* parser_ctx_ = nullptr;
    AVPacket*             pkt_        = nullptr;
    AVFrame*              hw_frame_   = nullptr;
    AVFrame*              sw_frame_   = nullptr;
    SwsContext*           sws_ctx_    = nullptr;
    cv::Mat               bgr_frame_;
    AVBufferRef*          hw_device_ctx_ = nullptr;
    enum AVHWDeviceType   hw_type_       = AV_HWDEVICE_TYPE_NONE;

    ros::NodeHandle pnh_;
    ros::Subscriber subscription_;
    ros::Publisher  publisher_;

    std::thread             publisher_thread_;
    std::queue<cv::Mat>     frame_publish_queue_;
    std::mutex              queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool>       stop_publisher_thread_{false};
    size_t                  max_queue_size_ = 10;

    void InitFFmpegDecoder() {
        hw_type_                 = AV_HWDEVICE_TYPE_CUDA;
        const char* decoder_name = "h264_cuvid";

        codec_ = avcodec_find_decoder_by_name(decoder_name);
        if (!codec_) {
            ROS_WARN(
                "Hardware decoder not available, falling back to software");
            hw_type_ = AV_HWDEVICE_TYPE_NONE;
            codec_   = avcodec_find_decoder(AV_CODEC_ID_H264);
            if (!codec_) {
                ROS_ERROR("No H.264 decoder available");
                return;
            }
        } else {
            ROS_INFO("Using hardware H.264 decoder (NVDEC)");
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            int err = av_hwdevice_ctx_create(&hw_device_ctx_, hw_type_, nullptr,
                                             nullptr, 0);
            if (err < 0) {
                ROS_WARN(
                    "Failed to create hardware device context, falling "
                    "back to software");
                hw_type_ = AV_HWDEVICE_TYPE_NONE;
                codec_   = avcodec_find_decoder(AV_CODEC_ID_H264);
                if (!codec_) {
                    ROS_ERROR("No H.264 decoder available");
                    return;
                }
            }
        }

        parser_ctx_ = av_parser_init(codec_->id);
        if (!parser_ctx_) {
            CleanupFFmpegDecoder();
            return;
        }

        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_) {
            CleanupFFmpegDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE && hw_device_ctx_) {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            codec_ctx_->get_format    = get_hw_format;
        }

        if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
            ROS_ERROR("Failed to open codec");
            CleanupFFmpegDecoder();
            return;
        }

        pkt_ = av_packet_alloc();
        if (!pkt_) {
            CleanupFFmpegDecoder();
            return;
        }

        hw_frame_ = av_frame_alloc();
        if (!hw_frame_) {
            CleanupFFmpegDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            sw_frame_ = av_frame_alloc();
            if (!sw_frame_) {
                CleanupFFmpegDecoder();
                return;
            }
        }
    }

    void PublisherThreadLoop() {
        while (!stop_publisher_thread_) {
            cv::Mat frame_to_publish;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_publish_queue_.empty() ||
                           stop_publisher_thread_;
                });

                if (stop_publisher_thread_ && frame_publish_queue_.empty()) {
                    break;
                }
                if (frame_publish_queue_.empty()) {
                    continue;
                }
                frame_to_publish = frame_publish_queue_.front();
                frame_publish_queue_.pop();
            }

            if (!frame_to_publish.empty()) {
                sensor_msgs::Image img_msg;
                std_msgs::Header   header;
                header.stamp    = ros::Time::now();
                header.frame_id = "camera_frame";
                cv_bridge::CvImage cv_image(header,
                                            sensor_msgs::image_encodings::BGR8,
                                            frame_to_publish);
                cv_image.toImageMsg(img_msg);
                publisher_.publish(img_msg);
            }
        }
    }

    void DecodeAndDisplayPacket(AVPacket* packet) {
        int ret = avcodec_send_packet(codec_ctx_, packet);
        if (ret < 0) {
            return;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return;
            } else if (ret < 0) {
                return;
            }

            AVFrame* frame_to_display = hw_frame_;

            if (hw_frame_->format == AV_PIX_FMT_CUDA ||
                hw_frame_->format == AV_PIX_FMT_VAAPI) {
                if (av_hwframe_transfer_data(sw_frame_, hw_frame_, 0) < 0) {
                    av_frame_unref(hw_frame_);
                    continue;
                }
                frame_to_display = sw_frame_;
            }

            if (!sws_ctx_ && frame_to_display->width > 0 &&
                frame_to_display->height > 0) {
                sws_ctx_ = sws_getContext(
                    frame_to_display->width, frame_to_display->height,
                    (AVPixelFormat)frame_to_display->format,
                    frame_to_display->width, frame_to_display->height,
                    AV_PIX_FMT_BGR24, SWS_POINT, nullptr, nullptr, nullptr);

                if (!sws_ctx_) {
                    av_frame_unref(hw_frame_);
                    if (frame_to_display == sw_frame_)
                        av_frame_unref(sw_frame_);
                    return;
                }
                bgr_frame_.create(frame_to_display->height,
                                  frame_to_display->width, CV_8UC3);
            }

            if (sws_ctx_ && !bgr_frame_.empty()) {
                uint8_t* dst_data[4] = {bgr_frame_.data, nullptr, nullptr,
                                        nullptr};
                int dst_linesize[4]  = {static_cast<int>(bgr_frame_.step[0]), 0,
                                        0, 0};

                sws_scale(sws_ctx_,
                          (const uint8_t* const*)frame_to_display->data,
                          frame_to_display->linesize, 0,
                          frame_to_display->height, dst_data, dst_linesize);

                cv::Mat frame_copy = bgr_frame_.clone();
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (frame_publish_queue_.size() < max_queue_size_) {
                        frame_publish_queue_.push(frame_copy);
                    }
                }
                queue_cv_.notify_one();
            }

            av_frame_unref(hw_frame_);
            if (frame_to_display == sw_frame_) {
                av_frame_unref(sw_frame_);
            }
        }
    }

    void CleanupFFmpegDecoder() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
            sw_frame_ = nullptr;
        }
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
            hw_frame_ = nullptr;
        }
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_close(codec_ctx_);
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        if (parser_ctx_) {
            av_parser_close(parser_ctx_);
            parser_ctx_ = nullptr;
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
            hw_device_ctx_ = nullptr;
        }
        codec_ = nullptr;
    }

    void compressed_image_callback(
        const sensor_msgs::CompressedImageConstPtr& msg) {
        if (msg->format != "h264") {
            return;
        }

        if (!codec_ctx_ || !parser_ctx_ || !pkt_ || !hw_frame_) {
            return;
        }

        const uint8_t* cur_data       = msg->data.data();
        size_t         remaining_size = msg->data.size();

        while (remaining_size > 0) {
            int bytes_parsed = av_parser_parse2(
                parser_ctx_, codec_ctx_, &pkt_->data, &pkt_->size, cur_data,
                static_cast<int>(remaining_size), AV_NOPTS_VALUE,
                AV_NOPTS_VALUE, 0);
            if (bytes_parsed < 0) {
                break;
            }
            cur_data += bytes_parsed;
            remaining_size -= bytes_parsed;

            if (pkt_->size > 0) {
                DecodeAndDisplayPacket(pkt_);
            }
        }
    }

public:
    H264DecoderNode(const ros::NodeHandle& pnh) : pnh_(pnh) {
        std::string subscribe_topic = pnh_.param<std::string>(
            "subscribe_topic", "/insta/image/compressed");
        std::string publish_topic =
            pnh_.param<std::string>("publish_topic", "/insta/image");

        subscription_ =
            pnh_.subscribe(subscribe_topic, 10,
                           &H264DecoderNode::compressed_image_callback, this);
        publisher_ = pnh_.advertise<sensor_msgs::Image>(publish_topic, 10);

        publisher_thread_ =
            std::thread(&H264DecoderNode::PublisherThreadLoop, this);

        InitFFmpegDecoder();

        ROS_INFO("H.264 Decoder Node initialized");
        ROS_INFO("Subscribing to: %s", subscribe_topic.c_str());
        ROS_INFO("Publishing to: %s", publish_topic.c_str());
    }

    ~H264DecoderNode() {
        stop_publisher_thread_ = true;
        queue_cv_.notify_one();
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
        CleanupFFmpegDecoder();
    }
};

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "decoder");
    ros::NodeHandle node("~");
    H264DecoderNode decoder(node);
    ros::spin();
    ros::shutdown();
    return 0;
}
