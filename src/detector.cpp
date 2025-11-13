
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector_v8s_p2/detector.h>
#include "rm_msgs/RadarTargetDetection.h"
#include "rm_msgs/RadarTargetDetectionArray.h"
#include "sensor_msgs/CompressedImage.h"

namespace rm_detector_v8
{
Detector::Detector()
{
  num_frame_ = 0;
  total_ms_ = 0;
  tracker_ = new rm_bytetrack::BYTETracker(20, 50);
}

void Detector::onInit()
{
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

  initalizeInfer();

  ros::NodeHandle nh_reconfig(nh_, nodelet_name_ + "_reconfig");
  server_ = new dynamic_reconfigure::Server<rm_detector_v8::dynamicConfig>(nh_reconfig);
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

void Detector::receiveFromCam(const sensor_msgs::CompressedImageConstPtr& image)
{
  if (num_frame_ > 1000)
  {
    num_frame_ = 0;
    total_ms_ = 0;
  }
  num_frame_++;
  auto start = std::chrono::system_clock::now();
  cv_image_ = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

  car_inferencer_.detect(cv_image_->image);

  if (!car_inferencer_.target_objects_.empty())
  {
    for (auto& object : car_inferencer_.target_objects_)
    {
      cv::Mat armor_cls_image = cv_image_->image(get_rect(cv_image_->image, object.bbox, car_inferencer_.kInputH_, car_inferencer_.kInputW_)).clone();
      armor_inferencer_.detect(armor_cls_image);
      if (armor_inferencer_.target_objects_.empty())
      {
        object.class_id = -1;
        object.conf = 0.1f;
        continue;
      }
//      int n = armor_inferencer_.target_objects_.size();
//      int max_index = 0;
//      float temp = armor_inferencer_.target_objects_[0].conf;
//      if (n > 1)
//      {
//        for (auto i = 1; i < n; ++i)
//        {
//          if (armor_inferencer_.target_objects_[i].conf > temp)
//          {
//            temp = armor_inferencer_.target_objects_[i].conf;
//            max_index = i;
//          }
//        }
//      }
//      object.class_id = armor_inferencer_.target_objects_[max_index].class_id;
      int min_index = 0;
      if (armor_inferencer_.target_objects_.size() > 1)
      {
        int n = armor_inferencer_.target_objects_.size();
        auto& init_armor = armor_inferencer_.target_objects_[0];
        float min = sqrt((init_armor.bbox[0] - IMAGE_CENTER_X) *
                             (init_armor.bbox[0] - IMAGE_CENTER_X) + (init_armor.bbox[1] - IMAGE_CENTER_Y) * (init_armor.bbox[1] - IMAGE_CENTER_Y));
        float temp = min;

        for (int i = 1; i < n; i++)
        {
          init_armor = armor_inferencer_.target_objects_[i];
          temp = sqrt((init_armor.bbox[0] - IMAGE_CENTER_X) *
                          (init_armor.bbox[0] - IMAGE_CENTER_X) + (init_armor.bbox[1] - IMAGE_CENTER_Y) * (init_armor.bbox[1] - IMAGE_CENTER_Y));
          if (min > temp)
          {
            armor_inferencer_.target_objects_[0] = init_armor;
            min = temp;
            min_index = i;
          }
        }
      }
      object.class_id = armor_inferencer_.target_objects_[0].class_id;
      object.conf = armor_inferencer_.target_objects_[0].conf;
      if ((target_is_red_ && object.class_id >= 0 && object.class_id <= 5) ||
          (!target_is_red_ && object.class_id >= 6 && object.class_id <= 11))
        continue;
    }

    std::vector<Object> objects;
    for (auto& targetObject : car_inferencer_.target_objects_)
    {
      Object object;
      object.rect = get_rect(cv_image_->image, targetObject.bbox, car_inferencer_.kInputH_, car_inferencer_.kInputW_);
      object.label = targetObject.class_id;
      object.prob = targetObject.conf;
      objects.push_back(object);
    }
    output_stracks_.clear();
    output_stracks_ = tracker_->update(objects);

    if (!output_stracks_.empty())
      publicMsg();
  }
  if (turn_on_image_)
  {
    cv::Mat img_clone = cv_image_->image.clone();
    draw_bbox(img_clone, car_inferencer_.target_objects_, car_inferencer_.kInputH_, car_inferencer_.kInputW_);
    camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_clone).toImageMsg());
    auto end = std::chrono::system_clock::now();
    total_ms_ = total_ms_ + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    for (int i = 0; i < output_stracks_.size(); i++)
    {
      std::vector<float> tlwh = output_stracks_[i].tlwh_;
      putText(cv_image_->image, cv::format("%d", output_stracks_[i].track_class_id_), cv::Point(tlwh[0], tlwh[1] - 5),
              0, 3, cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
      rectangle(cv_image_->image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cv::Scalar(255, 0, 0), 2);
    }
    putText(cv_image_->image,
            cv::format("frame: %d fps: %d num: %d", num_frame_, num_frame_ * 1000000 / total_ms_,
                       output_stracks_.size()),
            cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    camera_pub_track_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
  }
}

void Detector::dynamicCallback(rm_detector_v8::dynamicConfig& config)
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
  for (auto& output_strack : output_stracks_)
  {
    if ((output_strack.track_class_id_ == -1) ||
        (target_is_red_ && output_strack.track_class_id_ >= 0 && output_strack.track_class_id_ <= 5) ||
        (!target_is_red_ && output_strack.track_class_id_ >= 6 && output_strack.track_class_id_ <= 11))
      continue;
    rm_msgs::RadarTargetDetection data;
    data.id = output_strack.track_class_id_;
//    std::vector<float> temp;
//    float temp_w = output_strack.tlwh_[2] / 8.0f;
//    float temp_h = output_strack.tlwh_[3] / 8.0f;
//    temp.push_back(output_strack.tlwh_[0] + 3 * temp_w);
//    temp.push_back(output_strack.tlwh_[1] + 3 * temp_h);
//    temp.push_back(output_strack.tlwh_[0] + 5 * temp_w);
//    temp.push_back(output_strack.tlwh_[1] + 5 * temp_h);
    data.position.data.assign(output_strack.tlbr_.begin(), output_strack.tlbr_.end());
//    data.position.data.push_back(output_strack.tlwh_[0]);
//    data.position.data.push_back(output_strack.tlwh_[1]);
//    data.position.data.push_back(output_strack.tlwh_[0] + output_strack.tlwh_[2]);
//    data.position.data.push_back(output_strack.tlwh_[1] + output_strack.tlwh_[3]);

    array.detections.push_back(data);
  }
  roi_datas_pub_.publish(array);
}
}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector_v8::Detector, nodelet::Nodelet)