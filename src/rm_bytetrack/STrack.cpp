//
// Created by ywj on 24-1-14.
//

#include <utility>

#include "rm_bytetrack/STrack.h"

namespace rm_bytetrack
{
STrack::STrack(std::vector<float> tlwh, float score, int class_id)
{
  _tlwh_.resize(4);
  _tlwh_.assign(tlwh.begin(), tlwh.end());

  is_activated_ = false;
  track_class_id_ = class_id;
  state_ = TrackState::New;

  tlwh_.resize(4);
  tlbr_.resize(4);

  static_tlwh();
  static_tlbr();
  frame_id_ = 0;
  tracklet_len_ = 0;
  this->score_ = score;
  start_frame_ = 0;
}

STrack::~STrack() = default;

void STrack::activate(KalmanFilter& kalman_filter, int frame_id)
{
  this->kalman_filter_ = kalman_filter;
  //  this->track_class_id_ = STrack::next_id();

  std::vector<float> _tlwh_tmp(4);
  _tlwh_tmp[0] = this->_tlwh_[0];
  _tlwh_tmp[1] = this->_tlwh_[1];
  _tlwh_tmp[2] = this->_tlwh_[2];
  _tlwh_tmp[3] = this->_tlwh_[3];
  std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];
  auto mc = this->kalman_filter_.initiate(xyah_box);
  this->mean_ = mc.first;
  this->covariance_ = mc.second;

  static_tlwh();
  static_tlbr();

  this->tracklet_len_ = 0;
  this->state_ = TrackState::Tracked;
  if (frame_id == 1)
  {
    this->is_activated_ = true;
  }
  // this->is_activated = true;
  this->frame_id_ = frame_id;
  this->start_frame_ = frame_id;
}

void STrack::re_activate(STrack& new_track, int frame_id, bool new_id)
{
  std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh_);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];
  auto mc = this->kalman_filter_.update(this->mean_, this->covariance_, xyah_box);
  this->mean_ = mc.first;
  this->covariance_ = mc.second;

  static_tlwh();
  static_tlbr();

  this->tracklet_len_ = 0;
  this->state_ = TrackState::Tracked;
  this->is_activated_ = true;
  this->frame_id_ = frame_id;
  this->score_ = new_track.score_;
  //  if (new_id)
  //    this->track_class_id_ = next_id();
}

void STrack::update(STrack& new_track, int frame_id)
{
  this->frame_id_ = frame_id;
  this->tracklet_len_++;

  std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh_);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];

  auto mc = this->kalman_filter_.update(this->mean_, this->covariance_, xyah_box);
  this->mean_ = mc.first;
  this->covariance_ = mc.second;

  static_tlwh();
  static_tlbr();

  this->state_ = TrackState::Tracked;
  this->is_activated_ = true;

  //  this->score_ = new_track.score_;
  if (new_track.track_class_id_ != -1)
  {
    this->track_class_id_ = new_track.track_class_id_;
    this->score_ = new_track.score_;
  }

}

void STrack::static_tlwh()
{
  if (this->state_ == TrackState::New)
  {
    tlwh_[0] = _tlwh_[0];
    tlwh_[1] = _tlwh_[1];
    tlwh_[2] = _tlwh_[2];
    tlwh_[3] = _tlwh_[3];

    return;
  }

  tlwh_[0] = mean_[0];
  tlwh_[1] = mean_[1];
  tlwh_[2] = mean_[2];
  tlwh_[3] = mean_[3];

  tlwh_[2] *= tlwh_[3];
  tlwh_[0] -= tlwh_[2] / 2;
  tlwh_[1] -= tlwh_[3] / 2;
}

void STrack::static_tlbr()
{
  tlbr_.clear();
  tlbr_.assign(tlwh_.begin(), tlwh_.end());
  tlbr_[2] += tlbr_[0];
  tlbr_[3] += tlbr_[1];
}

std::vector<float> STrack::tlwh_to_xyah(std::vector<float> tlwh_tmp)
{
  std::vector<float> tlwh_output = std::move(tlwh_tmp);
  tlwh_output[0] += tlwh_output[2] / 2;
  tlwh_output[1] += tlwh_output[3] / 2;
  tlwh_output[2] /= tlwh_output[3];
  return tlwh_output;
}

std::vector<float> STrack::to_xyah() const
{
  return tlwh_to_xyah(tlwh_);
}

std::vector<float> STrack::tlbr_to_tlwh(std::vector<float>& tlbr)
{
  tlbr[2] -= tlbr[0];
  tlbr[3] -= tlbr[1];
  return tlbr;
}

void STrack::mark_lost()
{
  state_ = TrackState::Lost;
}

void STrack::mark_removed()
{
  state_ = TrackState::Removed;
}

int STrack::next_id()
{
  static int _count = 0;
  _count++;
  return _count;
}

int STrack::end_frame() const
{
  return this->frame_id_;
}

void STrack::multi_predict(std::vector<STrack*>& stracks, KalmanFilter& kalman_filter)
{
  for (auto& strack : stracks)
  {
    if (strack->state_ != TrackState::Tracked)
    {
      strack->mean_[7] = 0;
    }
    kalman_filter.predict(strack->mean_, strack->covariance_);
    strack->static_tlwh();
    strack->static_tlbr();
  }
}
}  // namespace rm_bytetrack