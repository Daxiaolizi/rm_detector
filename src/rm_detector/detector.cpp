/*
 * @Author: Daxiaolizi
 * @Date: 2025-11-06 13:53:29
 */
#include "rm_detector/detector.h"
#include "rm_msgs/RadarTargetDetection.h"
#include "rm_msgs/RadarTargetDetectionArray.h"
#include "sensor_msgs/CompressedImage.h"
#include <ros/ros.h>

namespace rm_detector 
{
    Detector::Detector()
    { 
    num_frame_ = 0;
    total_ms_ = 0;
    //track?
    }


void Detector::onInit()
{ 
//ros节点,發佈檢測結果
nh_ = getMTPrivateNodeHandle();
 nh_.getParam("car_model_path", car_model_path_);
  nh_.getParam("armor_model_path", armor_model_path_);
  nh_.getParam("car_input_h", car_inferencer_.kInputH_);
  nh_.getParam("car_input_w", car_inferencer_.kInputW_);
  nh_.getParam("car_conf_thresh", car_inferencer_.conf_thresh_);
  nh_.getParam("car_nms_thresh", car_inferencer_.nms_thresh_);
  nh_.getParam("armor_input_h", armor_inferencer_.kInputH_);
  nh_.getParam("armor_input_w", armor_inferencer_.kInputW_);
  nh_.getParam("armor_conf_thresh", armor_inferencer_.conf_thresh_);
  nh_.getParam("armor_nms_thresh", armor_inferencer_.nms_thresh_);
  nh_.getParam("car_batch_size", car_inferencer_.BatchSize_);
  nh_.getParam("armor_batch_size", armor_inferencer_.BatchSize_);

  nh_.getParam("camera_pub_name", camera_pub_name_);
  nh_.getParam("nodelet_name", nodelet_name_);
  nh_.getParam("target_is_red", target_is_red_);

  nh_.getParam("left_camera", left_camera_);

  initializeInfer();

  ros::NodeHandle nh_reconfig(nh_, nodelet_name_ + "_reconfig");
  server_ = new dynamic_reconfigure::Server<rm_detector::dynamicConfig>(nh_reconfig);
  callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
  server_->setCallback(callback_);

  if (left_camera_)  // TODO: Should we use the subscribeCamera function to receive camera info?
    camera_sub_ = nh_.subscribe("/hk_camera_left/image_raw/compressed", 1, &Detector::receiveFromCam, this);
  else
    camera_sub_ = nh_.subscribe("/hk_camera_right/image_raw/compressed", 1, &Detector::receiveFromCam, this);

  camera_pub_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_, 1);

  camera_pub_track_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_ + "_track_", 1);

  roi_datas_pub_ = nh_.advertise<rm_msgs::RadarTargetDetectionArray>("rm_radar/roi_datas", 10);
}
void Detector:: receiveFromCam(const sensor_msgs::CompressedImageConstPtr& image)
{
  // 接收圖像
  if (num_frame_ > 1000)
  {
    num_frame_ = 0;
    total_ms_ = 0;
  }
  num_frame_++;
  auto start = std::chrono::system_clock::now();
  cv_image_ = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

  car_inferencer_.detect(cv_image_->image);
  if (!car_inferencer_.target_objects_.empty() && !car_inferencer_.target_objects_[0].empty())
  {
    for (auto& object : car_inferencer_.target_objects_[0])
    {
      cv::Rect roi = get_rect(cv_image_->image, object.bbox, car_inferencer_.kInputH_, car_inferencer_.kInputW_);
      cv::Mat armor_cls_image = cv_image_->image(roi).clone();
      std::vector<cv::Mat> armor_batch = {armor_cls_image};
      armor_inferencer_.BatchSize_ = 1;
      armor_inferencer_.target_objects_.resize(1);
     
     armor_inferencer_.detect(armor_cls_image);
     if (armor_inferencer_.target_objects_[0].empty())
     {
      object.class_id = -1;
      object.conf = 0.1f;
      continue;
     }
     int min_index = 0;
     if (armor_inferencer_.target_objects_[0].size() > 1)
     {
       int n = armor_inferencer_.target_objects_[0].size();
       auto &init_armor = armor_inferencer_.target_objects_[0][0];
       float min = sqrt((init_armor.bbox[0] - IMAGE_CENTER_X) *
                            (init_armor.bbox[0] - IMAGE_CENTER_X) +
                        (init_armor.bbox[1] - IMAGE_CENTER_Y) * (init_armor.bbox[1] - IMAGE_CENTER_Y));
       float temp = min;

       for (int i = 1; i < n; i++)
       {
         init_armor = armor_inferencer_.target_objects_[0][i];
         temp = sqrt((init_armor.bbox[0] - IMAGE_CENTER_X) *
                         (init_armor.bbox[0] - IMAGE_CENTER_X) +
                     (init_armor.bbox[1] - IMAGE_CENTER_Y) * (init_armor.bbox[1] - IMAGE_CENTER_Y));
         if (min > temp)
         {
           armor_inferencer_.target_objects_[0][0] = init_armor;
           min = temp;
           min_index = i;
         }
       }
     }
     object.class_id = armor_inferencer_.target_objects_[0][0].class_id;
     object.conf = armor_inferencer_.target_objects_[0][0].conf;
    //  if ((target_is_red_ && object.class_id >= 0 && object.class_id <= 5) ||
    //      (!target_is_red_ && object.class_id >= 6 && object.class_id <= 11))
    //    continue;
    }

    std::vector<Object> objects;
    objects.clear();
    for (auto& car_targetObject : car_inferencer_.target_objects_[0])
    {
      cv::Rect car_rect = get_rect(cv_image_->image, car_targetObject.bbox, car_inferencer_.kInputH_, car_inferencer_.kInputW_);
      cv::Mat car_crop = cv_image_->image(car_rect).clone();

      armor_inferencer_.detect(car_crop);

      for (auto& armor_targetObject : armor_inferencer_.target_objects_[0]){
        Object object;

        cv::Rect armor_local = get_rect(car_crop, armor_targetObject.bbox, armor_inferencer_.kInputH_, armor_inferencer_.kInputW_);
        object.rect = armor_local + cv::Point(car_rect.x, car_rect.y);
        object.label = armor_targetObject.class_id;
        object.prob = armor_targetObject.conf;

        objects.push_back(object);
      }
    }
    output_stracks_.clear();
    output_stracks_ = tracker_->update(objects);
  }
  if (!output_stracks_.empty())
  {
    publicMsg();
  }

  if (turn_on_image_)
  {
    cv::Mat img_clone = cv_image_->image.clone();
    std::vector<cv::Mat> img_batch = {img_clone};
    std::vector<std::vector<Detection>> res_batch = {armor_inferencer_.target_objects_[0]};
    draw_bbox(img_batch, res_batch);

  camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_clone ).toImageMsg());

  auto end = std::chrono::system_clock::now();
  total_ms_ = total_ms_ + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  for (int i = 0; i < output_stracks_.size(); i++)
  {
    std::vector<float> tlwh = output_stracks_[i].tlwh_;
    cv::putText(cv_image_->image, cv::format("%d", output_stracks_[i].track_class_id_), cv::Point(tlwh[0], tlwh[1]- 5),
    0, 3, cv::Scalar(0, 0, 255), 4, cv::LINE_AA);

    cv::rectangle(cv_image_->image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cv::Scalar(255, 0, 0), 2);
  }
    cv::putText(cv_image_->image,
    cv::format("frame: %d fps: %d num: %ld", num_frame_, num_frame_ * 1000000 / total_ms_,
                       output_stracks_.size()),
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

void Detector::initializeInfer()
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
  for (auto &output_strack : output_stracks_)
  {
    if ((output_strack.track_class_id_ == -1) ||
        (target_is_red_ && output_strack.track_class_id_ >= 0 && output_strack.track_class_id_ <= 11) ||
        (!target_is_red_ && output_strack.track_class_id_ >= 0 && output_strack.track_class_id_ <= 11))
      continue;
    rm_msgs::RadarTargetDetection data;
    data.id = output_strack.track_class_id_;
    data.position.data.assign(output_strack.tlbr_.begin(), output_strack.tlbr_.end());

    array.detections.push_back(data);
  }
  roi_datas_pub_.publish(array);
}
}// namespace rm_detector

PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)