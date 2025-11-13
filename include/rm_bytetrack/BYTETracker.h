//
// Created by ywj on 24-1-14.
//

#ifndef RM_RADAR_BYTETRACK_BYTETRACKER_H
#define RM_RADAR_BYTETRACK_BYTETRACKER_H

#include "TensorRT_yolov8/types.h"
#include "rm_bytetrack/STrack.h"
#include "rm_bytetrack/dataType.h"
#include "rm_bytetrack/lapjv.h"

namespace rm_bytetrack
{
class BYTETracker
{
public:
  explicit BYTETracker(const int& frame_rate = 30, const int& track_buffer = 30, const float& track_thresh = 0.5,
                       const float& high_thresh = 0.6, const float& match_thresh = 0.8);
  ~BYTETracker();

  std::vector<STrack> update(const std::vector<Object>& objects);
  static cv::Scalar get_color(int idx);

private:
  static std::vector<STrack*> joint_stracks(std::vector<STrack*>& tlista, std::vector<STrack>& tlistb);
  static std::vector<STrack> joint_stracks(std::vector<STrack>& tlista, std::vector<STrack>& tlistb);

  static std::vector<STrack> sub_stracks(std::vector<STrack>& tlista, std::vector<STrack>& tlistb);
  void remove_duplicate_stracks(std::vector<STrack>& resa, std::vector<STrack>& resb, std::vector<STrack>& stracksa,
                                std::vector<STrack>& stracksb);

  void linear_assignment(std::vector<std::vector<float> >& cost_matrix, int cost_matrix_size, int cost_matrix_size_size,
                         float thresh, std::vector<std::vector<int> >& matches, std::vector<int>& unmatched_a,
                         std::vector<int>& unmatched_b);
  static std::vector<std::vector<float> > iou_distance(std::vector<STrack*>& atracks, std::vector<STrack>& btracks,
                                                       int& dist_size, int& dist_size_size);
  static std::vector<std::vector<float> > iou_distance(std::vector<STrack>& atracks, std::vector<STrack>& btracks);
  static std::vector<std::vector<float> > ious(std::vector<std::vector<float> >& atlbrs,
                                               std::vector<std::vector<float> >& btlbrs);

  static double lapjv(const std::vector<std::vector<float> >& cost, std::vector<int>& rowsol, std::vector<int>& colsol,
                      bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
  float track_thresh_;
  float high_thresh_;
  float match_thresh_;
  int frame_id_;
  int max_time_lost_;

  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  std::vector<STrack> removed_stracks_;
  KalmanFilter kalman_filter_;
};
}  // namespace rm_bytetrack

#endif  // RM_RADAR_BYTETRACK_BYTETRACKER_H
