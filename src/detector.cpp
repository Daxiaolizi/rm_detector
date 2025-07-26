
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>
#include "rm_msgs/RadarTargetDetection.h"
#include "rm_msgs/RadarTargetDetectionArray.h"
#include "sensor_msgs/CompressedImage.h"

namespace rm_detector
{
Detector::Detector()
{
  num_frame_ = 0;
  total_ms_ = 0;
//  tracker_ = new rm_bytetrack::BYTETracker(50, 100);
}

void Detector::onInit()
{
  nh_ = getMTPrivateNodeHandle();
  nh_.getParam("g_car_model_path", car_model_path_);
  nh_.getParam("g_armor_model_path", armor_model_path_);

  nh_.getParam("camera_pub_name", camera_pub_name_);
  nh_.getParam("nodelet_name", nodelet_name_);
  nh_.getParam("target_is_red", target_is_red_);

  nh_.getParam("left_camera", left_camera_);

  initalizeInfer();

  ros::NodeHandle nh_reconfig(nh_, nodelet_name_ + "_reconfig");
  server_ = new dynamic_reconfigure::Server<rm_detector::dynamicConfig>(nh_reconfig);
  callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
  server_->setCallback(callback_);

//  if (left_camera_)  // TODO: Should we use the subscribeCamera function to receive camera info?
//    camera_sub_ = nh_.subscribe("/hk_camera_right/image_raw/compressed", 1, &Detector::receiveFromCam, this);
//  else
//    camera_sub_ = nh_.subscribe("/hk_camera_left/image_raw/compressed", 1, &Detector::receiveFromCam, this);
  camera_sub_ = nh_.subscribe("/hk_stitched_image", 1, &Detector::receiveFromCam, this);

  camera_pub_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_, 1);

  camera_pub_track_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_ + "_track_", 1);

  roi_datas_pub_ = nh_.advertise<rm_msgs::RadarTargetDetectionArray>("rm_radar/roi_datas", 10);
}

void Detector::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
    ROS_INFO("Received image from /hk_stitched_image");

    if (num_frame_ > 1000)
  {
    num_frame_ = 0;
    total_ms_ = 0;
  }
  num_frame_++;
  auto start = std::chrono::system_clock::now();
  cv_image_ = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

  car_inferencer_.detect(cv_image_->image);

cv::Mat img_clone = cv_image_->image.clone();
  if (!car_inferencer_.target_objects_.empty())
  {
    for (auto& object : car_inferencer_.target_objects_)
    {
      cv::Mat armor_cls_image = cv_image_->image(get_rect(cv_image_->image, object.bbox)).clone();

      armor_inferencer_.detect(armor_cls_image);
      if (armor_inferencer_.target_objects_.empty())
      {
        object.class_id = -1;
        object.conf = 0.1f;
        continue;
      }
      int n = armor_inferencer_.target_objects_.size();
      int max_index = 0;
      float temp = armor_inferencer_.target_objects_[0].conf;
      if (n > 1)
      {
        for (auto i = 1; i < n; ++i)
        {
          if (armor_inferencer_.target_objects_[i].conf > temp)
          {
            temp = armor_inferencer_.target_objects_[i].conf;
            max_index = i;
          }
        }
      }
      object.class_id = armor_inferencer_.target_objects_[max_index].class_id;
      if ((target_is_red_ && object.class_id >= 0 && object.class_id <= 5) ||
          (!target_is_red_ && object.class_id >= 6 && object.class_id <= 11))
        continue;
      object.conf = temp;
        // Adjust armor bounding box coordinates to original image
        cv::Rect car_rect = get_rect(cv_image_->image, object.bbox);
        for (auto& armor_object : armor_inferencer_.target_objects_)
        {
            cv::Rect armor_rect = get_rect(armor_cls_image, armor_object.bbox);
            // Translate armor bounding box to original image coordinates
            armor_rect.x += car_rect.x;
            armor_rect.y += car_rect.y;
            armor_object.bbox[0] =  static_cast<float>(armor_rect.x + armor_rect.width / 2.0f);
            armor_object.bbox[1] =  static_cast<float>(armor_rect.y + armor_rect.height / 2.0f);
            armor_object.bbox[2] =  static_cast<float>(armor_rect.width);
            armor_object.bbox[3] =  static_cast<float>(armor_rect.height);
        }
        // Draw armor bounding boxes
        draw_bbox(img_clone, armor_inferencer_.target_objects_);
    }

//    std::vector<Object> objects;
//    for (auto& targetObject : car_inferencer_.target_objects_)
//    {
//      Object object;
//      object.rect = get_rect(cv_image_->image, targetObject.bbox);
//      object.label = targetObject.class_id;
//      object.prob = targetObject.conf;
//      objects.push_back(object);
//    }
//    output_stracks_.clear();
//    output_stracks_ = tracker_->update(objects);

//    if (!output_stracks_.empty())
      publicMsg();
  }
  if (turn_on_image_)
  {
    cv::Mat img_clone = cv_image_->image.clone();
    draw_bbox(img_clone, car_inferencer_.target_objects_);
    camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_clone).toImageMsg());
    auto end = std::chrono::system_clock::now();
    total_ms_ = total_ms_ + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    for (auto& targetObject : car_inferencer_.target_objects_)
    {
//      std::vector<float> tlwh = output_stracks_[i].tlwh_;
        if (targetObject.class_id == -1 || (target_is_red_ && targetObject.class_id >= 0 && targetObject.class_id <= 5) ||
                                           (!target_is_red_ && targetObject.class_id >= 6 && targetObject.class_id <= 11))
            continue;
        cv::Rect rect = get_rect(cv_image_->image, targetObject.bbox);
      putText(cv_image_->image, cv::format("%d", targetObject.class_id), cv::Point(rect.x, rect.y - 5),
              0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
      rectangle(cv_image_->image, rect, cv::Scalar(255, 0, 0), 2);
    }
    putText(cv_image_->image,
            cv::format("frame: %d fps: %d num: %lu", num_frame_, num_frame_ * 1000000 / total_ms_,
                       car_inferencer_.target_objects_.size()),
            cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    camera_pub_track_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
  }
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
  car_inferencer_.conf_thresh_ = config.g_car_conf_thresh;
  car_inferencer_.nms_thresh_ = config.g_car_nms_thresh;
  armor_inferencer_.conf_thresh_ = config.g_armor_conf_thresh;
  armor_inferencer_.nms_thresh_ = config.g_armor_nms_thresh;
  turn_on_image_ = config.g_turn_on_image;
  ROS_INFO("Settings have been seted");
}

void Detector::initalizeInfer()
{
  cudaSetDevice(kGpuId);
  car_inferencer_.init(car_model_path_, gLogger_);
  armor_inferencer_.init(armor_model_path_, gLogger_);
}

Detector::~Detector()
{
  this->roi_array_.detections.clear();
}

void Detector::publicMsg()
{
  rm_msgs::RadarTargetDetectionArray array;
  array.header.stamp = ros::Time::now();
  for (auto& targetObject : car_inferencer_.target_objects_)
  {
    if ((targetObject.class_id == -1) ||
        (target_is_red_ && targetObject.class_id >= 0 && targetObject.class_id <= 5) ||
        (!target_is_red_ && targetObject.class_id >= 6 && targetObject.class_id <= 11))
      continue;
    rm_msgs::RadarTargetDetection data;
    data.id = targetObject.class_id;
    //    std::vector<float> temp;
    //    float temp_w = output_strack.tlwh_[2] / 8.0f;
    //    float temp_h = output_strack.tlwh_[3] / 8.0f;
    //    temp.push_back(output_strack.tlwh_[0] + 3 * temp_w);
    //    temp.push_back(output_strack.tlwh_[1] + 3 * temp_h);
    //    temp.push_back(output_strack.tlwh_[0] + 5 * temp_w);
    //    temp.push_back(output_strack.tlwh_[1] + 5 * temp_h);
      cv::Rect rect = get_rect(cv_image_->image, targetObject.bbox);
      std::vector<float> tlbr = {static_cast<float>(rect.x), static_cast<float>(rect.y),
                                 static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height)};
    data.position.data.assign(tlbr.begin(), tlbr.end());
    //    data.position.data.push_back(output_strack.tlwh_[0]);
    //    data.position.data.push_back(output_strack.tlwh_[1]);
    //    data.position.data.push_back(output_strack.tlwh_[0] + output_strack.tlwh_[2]);
    //    data.position.data.push_back(output_strack.tlwh_[1] + output_strack.tlwh_[3]);

    array.detections.push_back(data);
  }
  roi_datas_pub_.publish(array);
}
}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)