//
// Created by ywj on 24-1-14.
//

#ifndef RM_RADAR_BYTETRACK_STRACK_H
#define RM_RADAR_BYTETRACK_STRACK_H

#include <opencv2/opencv.hpp>
#include "rm_bytetrack/kalmanFilter.h"

enum TrackState
{
  New = 0,
  Tracked,
  Lost,
  Removed
};

namespace rm_bytetrack
{
class STrack
{
public:
  STrack(std::vector<float> tlwh_, float score, int class_id);
  ~STrack();

  std::vector<float> static tlbr_to_tlwh(std::vector<float>& tlbr);
  void static multi_predict(std::vector<STrack*>& stracks, KalmanFilter& kalman_filter);
  void static_tlwh();
  void static_tlbr();
  static std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);
  std::vector<float> to_xyah() const;
  void mark_lost();
  void mark_removed();
  static int next_id();
  int end_frame() const;

  void activate(KalmanFilter& kalman_filter, int frame_id);
  void re_activate(STrack& new_track, int frame_id, bool new_id = false);
  void update(STrack& new_track, int frame_id);

public:
  bool is_activated_;
  int track_class_id_;
  int state_;
  //        int class_id;

  std::vector<float> _tlwh_;
  std::vector<float> tlwh_;
  std::vector<float> tlbr_;
  int frame_id_;
  int tracklet_len_;
  int start_frame_;

  KAL_MEAN mean_;
  KAL_COVA covariance_;
  float score_;

private:
  KalmanFilter kalman_filter_;
};
}  // namespace rm_bytetrack

#endif  // RM_RADAR_BYTETRACK_STRACK_H
