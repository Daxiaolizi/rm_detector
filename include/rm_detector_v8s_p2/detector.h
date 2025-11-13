//
// Created by yamabuki on 2022/4/18.
//
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>
#include <std_msgs/Float32MultiArray.h>
// #include <rm_msgs/RadarTargetDetectionArray.h>
// #include <rm_msgs/RadarTargetDetection.h>
#include "rm_msgs/RadarTargetDetection.h"
#include "rm_msgs/RadarTargetDetectionArray.h"
#include <dynamic_reconfigure/server.h>
#include "rm_detector_v8/dynamicConfig.h"
#include <sensor_msgs/CompressedImage.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include "rm_detector_v8s_p2/inferencer.h"
#include "rm_bytetrack/BYTETracker.h"

// static Logger gLogger_;
#define IMAGE_CENTER_X 1536
#define IMAGE_CENTER_Y 1024

namespace rm_detector_v8
{
class Detector : public nodelet::Nodelet
{
public:
  Detector();
  ~Detector() override;

  void onInit() override;
  void receiveFromCam(const sensor_msgs::CompressedImageConstPtr& image);
  void publicMsg();
  void initalizeInfer();
  void dynamicCallback(rm_detector_v8::dynamicConfig& config);

  cv_bridge::CvImagePtr cv_image_;

  std::vector<cv::Point2f> roi_point_vec_;
  cv::Point2f roi_data_point_r_;
  cv::Point2f roi_data_point_l_;

  std::string car_model_path_;
  std::string armor_model_path_;

  bool turn_on_image_;
  dynamic_reconfigure::Server<rm_detector_v8::dynamicConfig>* server_;
  dynamic_reconfigure::Server<rm_detector_v8::dynamicConfig>::CallbackType callback_;

  std::string camera_pub_name_;
  std::string nodelet_name_;

  std::string roi_data1_name_;  // hero
  std::string roi_data2_name_;  // engineer
  std::string roi_data3_name_;  // standard
  std::string roi_data4_name_;  // standard
  std::string roi_data5_name_;  // standard
  std::string roi_data6_name_;  // sentry
  std::string roi_data7_name_;  // our sentry

  bool target_is_red_;
  bool left_camera_;

  Inferencer car_inferencer_;
  Inferencer armor_inferencer_;

  ros::NodeHandle nh_;
  Logger gLogger_;

private:
  unsigned int num_frame_;
  unsigned int total_ms_;
  rm_bytetrack::BYTETracker* tracker_ = nullptr;
  std::vector<rm_bytetrack::STrack> output_stracks_;

  ros::Publisher camera_pub_;
  ros::Publisher camera_pub_track_;
  ros::Subscriber camera_sub_;
  ros::Publisher roi_datas_pub_;

  rm_msgs::RadarTargetDetectionArray roi_array_{};
};
}  // namespace rm_detector