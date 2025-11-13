//
// Created by ywj on 24-1-14.
//

#include "rm_bytetrack/BYTETracker.h"
#include "rm_bytetrack/STrack.h"

#include <fstream>

namespace rm_bytetrack
{
BYTETracker::BYTETracker(const int& frame_rate, const int& track_buffer, const float& track_thresh,
                         const float& high_thresh, const float& match_thresh)
{
  this->track_thresh_ = track_thresh;
  this->high_thresh_ = high_thresh;
  this->match_thresh_ = match_thresh;

  frame_id_ = 0;
  max_time_lost_ = int(frame_rate / 30.0 * track_buffer);
  std::cout << "Init ByteTrack!" << std::endl;
}

BYTETracker::~BYTETracker() = default;

std::vector<STrack> BYTETracker::update(const std::vector<Object>& objects)
{
  //  std::cout << "step 1" << std::endl;
  ////////////////// Step 1: Get detections //////////////////
  this->frame_id_++;
  std::vector<STrack> activated_stracks;
  std::vector<STrack> refind_stracks;
  std::vector<STrack> removed_stracks;
  std::vector<STrack> lost_stracks;
  std::vector<STrack> detections;
  std::vector<STrack> detections_low;

  std::vector<STrack> detections_cp;
  std::vector<STrack> tracked_stracks_swap;
  std::vector<STrack> resa, resb;
  std::vector<STrack> output_stracks;

  std::vector<STrack*> unconfirmed;
  std::vector<STrack*> tracked_stracks;
  std::vector<STrack*> strack_pool;
  std::vector<STrack*> r_tracked_stracks;

  if (!objects.empty())
  {
    for (auto& object : objects)
    {
      std::vector<float> tlbr_;
      tlbr_.resize(4);
      tlbr_[0] = object.rect.x;
      tlbr_[1] = object.rect.y;
      tlbr_[2] = object.rect.x + object.rect.width;
      tlbr_[3] = object.rect.y + object.rect.height;

      float score = object.prob;

      STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, object.label);

      if (score >= track_thresh_)
      {
        detections.push_back(strack);
      }
      else
      {
        detections_low.push_back(strack);
      }
    }
  }

  // Add newly detected tracklets to tracked_stracks
  for (auto& tracked_strack : this->tracked_stracks_)
  {
    if (!tracked_strack.is_activated_)
      unconfirmed.push_back(&tracked_strack);
    else
      tracked_stracks.push_back(&tracked_strack);
  }
  //  std::cout << "step 2" << std::endl;
  ////////////////// Step 2: First association, with IoU //////////////////
  strack_pool = joint_stracks(tracked_stracks, this->lost_stracks_);
  STrack::multi_predict(strack_pool, this->kalman_filter_);

  std::vector<std::vector<float>> dists;
  int dist_size = 0, dist_size_size = 0;
  dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

  std::vector<std::vector<int>> matches;
  std::vector<int> u_track, u_detection;
  linear_assignment(dists, dist_size, dist_size_size, match_thresh_, matches, u_track, u_detection);

  for (auto& matche : matches)
  {
    STrack* track = strack_pool[matche[0]];
    STrack* det = &detections[matche[1]];
    if (track->state_ == TrackState::Tracked)
    {
      track->update(*det, this->frame_id_);
      activated_stracks.push_back(*track);
    }
    else
    {
      track->re_activate(*det, this->frame_id_, false);
      refind_stracks.push_back(*track);
    }
  }

  //  std::cout << "step 3" << std::endl;
  ////////////////// Step 3: Second association, using low score dets //////////////////
  for (int i : u_detection)
  {
    detections_cp.push_back(detections[i]);
  }
  detections.clear();
  detections.assign(detections_low.begin(), detections_low.end());

  for (int i : u_track)
  {
    if (strack_pool[i]->state_ == TrackState::Tracked)
    {
      r_tracked_stracks.push_back(strack_pool[i]);
    }
  }

  dists.clear();
  dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

  matches.clear();
  u_track.clear();
  u_detection.clear();
  linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

  for (auto& matche : matches)
  {
    STrack* track = r_tracked_stracks[matche[0]];
    STrack* det = &detections[matche[1]];
    if (track->state_ == TrackState::Tracked)
    {
      track->update(*det, this->frame_id_);
      activated_stracks.push_back(*track);
    }
    else
    {
      track->re_activate(*det, this->frame_id_, false);
      refind_stracks.push_back(*track);
    }
  }

  for (int i : u_track)
  {
    STrack* track = r_tracked_stracks[i];
    if (track->state_ != TrackState::Lost)
    {
      track->mark_lost();
      lost_stracks.push_back(*track);
    }
  }

  // Deal with unconfirmed tracks, usually tracks with only one beginning frame
  detections.clear();
  detections.assign(detections_cp.begin(), detections_cp.end());

  dists.clear();
  dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

  matches.clear();
  std::vector<int> u_unconfirmed;
  u_detection.clear();
  linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

  for (auto& matche : matches)
  {
    unconfirmed[matche[0]]->update(detections[matche[1]], this->frame_id_);
    activated_stracks.push_back(*unconfirmed[matche[0]]);
  }

  for (int i : u_unconfirmed)
  {
    STrack* track = unconfirmed[i];
    track->mark_removed();
    removed_stracks.push_back(*track);
  }

  //  std::cout << "step 4" << std::endl;
  ////////////////// Step 4: Init new stracks //////////////////
  for (int i : u_detection)
  {
    STrack* track = &detections[i];
    if (track->score_ < this->high_thresh_)
      continue;
    track->activate(this->kalman_filter_, this->frame_id_);
    activated_stracks.push_back(*track);
  }

  //  std::cout << "step 5" << std::endl;
  ////////////////// Step 5: Update state //////////////////
  for (auto& lost_strack : this->lost_stracks_)
  {
    if (this->frame_id_ - lost_strack.end_frame() > this->max_time_lost_)
    {
      lost_strack.mark_removed();
      removed_stracks.push_back(lost_strack);
    }
  }

  for (auto& tracked_strack : this->tracked_stracks_)
  {
    if (tracked_strack.state_ == TrackState::Tracked)
    {
      tracked_stracks_swap.push_back(tracked_strack);
    }
  }
  this->tracked_stracks_.clear();
  this->tracked_stracks_.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

  this->tracked_stracks_ = joint_stracks(this->tracked_stracks_, activated_stracks);
  this->tracked_stracks_ = joint_stracks(this->tracked_stracks_, refind_stracks);

  this->lost_stracks_ = sub_stracks(this->lost_stracks_, this->tracked_stracks_);
  for (const auto& lost_strack : lost_stracks)
  {
    this->lost_stracks_.push_back(lost_strack);
  }

  this->lost_stracks_ = sub_stracks(this->lost_stracks_, this->removed_stracks_);
  for (const auto& removed_strack : removed_stracks)
  {
    this->removed_stracks_.push_back(removed_strack);
  }

  remove_duplicate_stracks(resa, resb, this->tracked_stracks_, this->lost_stracks_);

  this->tracked_stracks_.clear();
  this->tracked_stracks_.assign(resa.begin(), resa.end());
  this->lost_stracks_.clear();
  this->lost_stracks_.assign(resb.begin(), resb.end());

  for (auto& tracked_strack : this->tracked_stracks_)
  {
    if (tracked_strack.is_activated_)
    {
      output_stracks.push_back(tracked_strack);
    }
  }
  return output_stracks;
}

std::vector<STrack*> BYTETracker::joint_stracks(std::vector<STrack*>& tlista, std::vector<STrack>& tlistb)
{
  std::map<int, int> exists;
  std::vector<STrack*> res;
  for (auto& i : tlista)
  {
    exists.insert(std::pair<int, int>(i->track_class_id_, 1));
    res.push_back(i);
  }
  for (auto& i : tlistb)
  {
    int tid = i.track_class_id_;
    if (!exists[tid] || exists.count(tid) == 0)
    {
      exists[tid] = 1;
      res.push_back(&i);
    }
  }
  return res;
}

std::vector<STrack> BYTETracker::joint_stracks(std::vector<STrack>& tlista, std::vector<STrack>& tlistb)
{
  std::map<int, int> exists;
  std::vector<STrack> res;
  for (auto& i : tlista)
  {
    exists.insert(std::pair<int, int>(i.track_class_id_, 1));
    res.push_back(i);
  }
  for (auto& i : tlistb)
  {
    int tid = i.track_class_id_;
    if (!exists[tid] || exists.count(tid) == 0)
    {
      exists[tid] = 1;
      res.push_back(i);
    }
  }
  return res;
}

std::vector<STrack> BYTETracker::sub_stracks(std::vector<STrack>& tlista, std::vector<STrack>& tlistb)
{
  std::map<int, STrack> stracks;
  for (auto& i : tlista)
  {
    stracks.insert(std::pair<int, STrack>(i.track_class_id_, i));
  }
  for (auto& i : tlistb)
  {
    int tid = i.track_class_id_;
    if (stracks.count(tid) != 0)
    {
      stracks.erase(tid);
    }
  }

  std::vector<STrack> res;
  std::map<int, STrack>::iterator it;
  for (it = stracks.begin(); it != stracks.end(); ++it)
  {
    res.push_back(it->second);
  }

  return res;
}

void BYTETracker::remove_duplicate_stracks(std::vector<STrack>& resa, std::vector<STrack>& resb,
                                           std::vector<STrack>& stracksa, std::vector<STrack>& stracksb)
{
  std::vector<std::vector<float>> pdist = iou_distance(stracksa, stracksb);
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < pdist.size(); i++)
  {
    for (int j = 0; j < pdist[i].size(); j++)
    {
      if (pdist[i][j] < 0.15)
      {
        pairs.emplace_back(i, j);
      }
    }
  }

  std::vector<int> dupa, dupb;
  for (auto& pair : pairs)
  {
    int timep = stracksa[pair.first].frame_id_ - stracksa[pair.first].start_frame_;
    int timeq = stracksb[pair.second].frame_id_ - stracksb[pair.second].start_frame_;
    if (timep > timeq)
      dupb.push_back(pair.second);
    else
      dupa.push_back(pair.first);
  }

  for (int i = 0; i < stracksa.size(); i++)
  {
    auto iter = find(dupa.begin(), dupa.end(), i);
    if (iter == dupa.end())
    {
      resa.push_back(stracksa[i]);
    }
  }

  for (int i = 0; i < stracksb.size(); i++)
  {
    auto iter = find(dupb.begin(), dupb.end(), i);
    if (iter == dupb.end())
    {
      resb.push_back(stracksb[i]);
    }
  }
}

void BYTETracker::linear_assignment(std::vector<std::vector<float>>& cost_matrix, int cost_matrix_size,
                                    int cost_matrix_size_size, float thresh, std::vector<std::vector<int>>& matches,
                                    std::vector<int>& unmatched_a, std::vector<int>& unmatched_b)
{
  if (cost_matrix.empty())
  {
    for (int i = 0; i < cost_matrix_size; i++)
    {
      unmatched_a.push_back(i);
    }
    for (int i = 0; i < cost_matrix_size_size; i++)
    {
      unmatched_b.push_back(i);
    }
    return;
  }

  std::vector<int> rowsol;
  std::vector<int> colsol;
  float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
  for (int i = 0; i < rowsol.size(); i++)
  {
    if (rowsol[i] >= 0)
    {
      std::vector<int> match;
      match.push_back(i);
      match.push_back(rowsol[i]);
      matches.push_back(match);
    }
    else
    {
      unmatched_a.push_back(i);
    }
  }

  for (int i = 0; i < colsol.size(); i++)
  {
    if (colsol[i] < 0)
    {
      unmatched_b.push_back(i);
    }
  }
}

std::vector<std::vector<float>> BYTETracker::ious(std::vector<std::vector<float>>& atlbrs,
                                                  std::vector<std::vector<float>>& btlbrs)
{
  std::vector<std::vector<float>> ious;
  if (atlbrs.size() * btlbrs.size() == 0)
    return ious;

  ious.resize(atlbrs.size());
  for (auto& iou : ious)
  {
    iou.resize(btlbrs.size());
  }

  // bbox_ious
  for (int k = 0; k < btlbrs.size(); k++)
  {
    std::vector<float> ious_tmp;
    float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
    for (int n = 0; n < atlbrs.size(); n++)
    {
      float iw = std::min(atlbrs[n][2], btlbrs[k][2]) - std::max(atlbrs[n][0], btlbrs[k][0]) + 1;
      if (iw > 0)
      {
        float ih = std::min(atlbrs[n][3], btlbrs[k][3]) - std::max(atlbrs[n][1], btlbrs[k][1]) + 1;
        if (ih > 0)
        {
          float ua = (atlbrs[n][2] - atlbrs[n][0] + 1) * (atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
          ious[n][k] = iw * ih / ua;
        }
        else
        {
          ious[n][k] = 0.0;
        }
      }
      else
      {
        ious[n][k] = 0.0;
      }
    }
  }

  return ious;
}

std::vector<std::vector<float>> BYTETracker::iou_distance(std::vector<STrack*>& atracks, std::vector<STrack>& btracks,
                                                          int& dist_size, int& dist_size_size)
{
  std::vector<std::vector<float>> cost_matrix;
  if (atracks.size() * btracks.size() == 0)
  {
    dist_size = atracks.size();
    dist_size_size = btracks.size();
    return cost_matrix;
  }
  std::vector<std::vector<float>> atlbrs, btlbrs;
  for (auto& atrack : atracks)
  {
    atlbrs.push_back(atrack->tlbr_);
  }
  for (auto& btrack : btracks)
  {
    btlbrs.push_back(btrack.tlbr_);
  }

  dist_size = atracks.size();
  dist_size_size = btracks.size();

  std::vector<std::vector<float>> _ious = ious(atlbrs, btlbrs);

  for (auto& i : _ious)
  {
    std::vector<float> _iou;
    for (float j : i)
    {
      _iou.push_back(1 - j);
    }
    cost_matrix.push_back(_iou);
  }

  return cost_matrix;
}

std::vector<std::vector<float>> BYTETracker::iou_distance(std::vector<STrack>& atracks, std::vector<STrack>& btracks)
{
  std::vector<std::vector<float>> atlbrs, btlbrs;
  for (auto& atrack : atracks)
  {
    atlbrs.push_back(atrack.tlbr_);
  }
  for (auto& btrack : btracks)
  {
    btlbrs.push_back(btrack.tlbr_);
  }

  std::vector<std::vector<float>> _ious = ious(atlbrs, btlbrs);
  std::vector<std::vector<float>> cost_matrix;
  for (auto& i : _ious)
  {
    std::vector<float> _iou;
    for (float j : i)
    {
      _iou.push_back(1 - j);
    }
    cost_matrix.push_back(_iou);
  }

  return cost_matrix;
}

double BYTETracker::lapjv(const std::vector<std::vector<float>>& cost, std::vector<int>& rowsol,
                          std::vector<int>& colsol, bool extend_cost, float cost_limit, bool return_cost)
{
  std::vector<std::vector<float>> cost_c;
  cost_c.assign(cost.begin(), cost.end());

  std::vector<std::vector<float>> cost_c_extended;

  int n_rows = cost.size();
  int n_cols = cost[0].size();
  rowsol.resize(n_rows);
  colsol.resize(n_cols);

  int n = 0;
  if (n_rows == n_cols)
  {
    n = n_rows;
  }
  else
  {
    if (!extend_cost)
    {
      std::cout << "set extend_cost=True" << std::endl;
      system("pause");
      exit(0);
    }
  }

  if (extend_cost || cost_limit < LONG_MAX)
  {
    n = n_rows + n_cols;
    cost_c_extended.resize(n);
    for (auto& i : cost_c_extended)
      i.resize(n);

    if (cost_limit < LONG_MAX)
    {
      for (auto& i : cost_c_extended)
      {
        for (float& j : i)
        {
          j = cost_limit / 2.0;
        }
      }
    }
    else
    {
      float cost_max = -1;
      for (auto& i : cost_c)
      {
        for (float j : i)
        {
          if (j > cost_max)
            cost_max = j;
        }
      }
      for (auto& i : cost_c_extended)
      {
        for (float& j : i)
        {
          j = cost_max + 1;
        }
      }
    }

    for (int i = n_rows; i < cost_c_extended.size(); i++)
    {
      for (int j = n_cols; j < cost_c_extended[i].size(); j++)
      {
        cost_c_extended[i][j] = 0;
      }
    }
    for (int i = 0; i < n_rows; i++)
    {
      for (int j = 0; j < n_cols; j++)
      {
        cost_c_extended[i][j] = cost_c[i][j];
      }
    }

    cost_c.clear();
    cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
  }

  double** cost_ptr;
  cost_ptr = new double*[sizeof(double*) * n];
  for (int i = 0; i < n; i++)
    cost_ptr[i] = new double[sizeof(double) * n];

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      cost_ptr[i][j] = cost_c[i][j];
    }
  }

  int* x_c = new int[sizeof(int) * n];
  int* y_c = new int[sizeof(int) * n];

  int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
  if (ret != 0)
  {
    std::cout << "Calculate Wrong!" << std::endl;
    system("pause");
    exit(0);
  }

  double opt = 0.0;

  if (n != n_rows)
  {
    for (int i = 0; i < n; i++)
    {
      if (x_c[i] >= n_cols)
        x_c[i] = -1;
      if (y_c[i] >= n_rows)
        y_c[i] = -1;
    }
    for (int i = 0; i < n_rows; i++)
    {
      rowsol[i] = x_c[i];
    }
    for (int i = 0; i < n_cols; i++)
    {
      colsol[i] = y_c[i];
    }

    if (return_cost)
    {
      for (int i = 0; i < rowsol.size(); i++)
      {
        if (rowsol[i] != -1)
        {
          // cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
          opt += cost_ptr[i][rowsol[i]];
        }
      }
    }
  }
  else if (return_cost)
  {
    for (int i = 0; i < rowsol.size(); i++)
    {
      opt += cost_ptr[i][rowsol[i]];
    }
  }

  for (int i = 0; i < n; i++)
  {
    delete[] cost_ptr[i];
  }
  delete[] cost_ptr;
  delete[] x_c;
  delete[] y_c;

  return opt;
}

cv::Scalar BYTETracker::get_color(int idx)
{
  idx += 3;
  return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}
}  // namespace rm_bytetrack