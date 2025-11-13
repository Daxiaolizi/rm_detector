/*
 * @Author: Daxiaolizi
 * @Date: 2025-11-06 17:15:53
 */
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>
#include <std_msgs/Float32MultiArray.h>
#include "rm_msgs/RadarTargetDetection.h"
#include "rm_msgs/RadarTargetDetectionArray.h"
#include <dynamic_reconfigure/server.h>
#include "rm_detector/dynamicConfig.h"
#include <sensor_msgs/CompressedImage.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include "rm_detector/inferencer.h"
#include "rm_bytetrack/BYTETracker.h"
#include "Infer_yolov12/types.h"
//接收圖像
//硬编码Logger glogger
// #define IMAGE_CENTER_X 1536
// #define IMAGE_CENTER_Y 1024

namespace rm_detector
{ 

class Detector : public nodelet::Nodelet
{ 
public:
Detector();
~Detector() override;

void onInit() override;
void receiveFromCam(const sensor_msgs::CompressedImageConstPtr& image);
void publicMsg();
void initializeInfer();
void dynamicCallback(rm_detector::dynamicConfig& config);

cv_bridge::CvImagePtr cv_image_;

//infer settings
std::vector<cv::Point2f> roi_point_vec;
cv::Point2f roi_data_point_r_;
cv::Point2f roi_data_point_l_;

std::string car_model_path_;
std::string armor_model_path_;

bool turn_on_image_;
dynamic_reconfigure::Server<rm_detector::dynamicConfig>* server_;
dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback_;

std::string camera_pub_name_;
std::string nodelet_name_;

std::string roi_data1_name_;//our hero
std::string roi_data2_name_;//our engineer
std::string roi_data3_name_;//our standard
std::string roi_data4_name_;//our standard
std::string roi_data5_name_;//our sentry
std::string roi_data6_name_;//hero
std::string roi_data7_name_;//engineer
std::string roi_data8_name_;//standard
std::string roi_data9_name_;//standard
std::string roi_data10_name_;//sentry

bool target_is_red_;
bool left_camera_;

Inferencer car_inferencer_;
Inferencer armor_inferencer_;

ros::NodeHandle nh_;
Logger gLogger_;

private:
static constexpr int IMAGE_CENTER_X = 1536;
static constexpr int IMAGE_CENTER_Y = 1024;
unsigned int num_frame_;
unsigned int total_ms_;
rm_bytetrack::BYTETracker *tracker_ = nullptr;
std::vector<rm_bytetrack::STrack> output_stracks_;

ros::Publisher camera_pub_;
ros::Publisher camera_pub_track_;
ros::Subscriber camera_sub_;
ros::Publisher roi_datas_pub_;

rm_msgs::RadarTargetDetectionArray roi_array_{};
};

}