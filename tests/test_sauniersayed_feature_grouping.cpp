#include <gtest/gtest.h>
#include <vector>
#include <cv.h>
#include <iostream>
#include "SaunierSayed_feature_grouping.h"

using std::vector;
using namespace cv;

namespace ss = SaunierSayed;

class SSTrackManagerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
      track_manager_ = ss::TrackManager(2, 4, 20);

      // points detected at time t
      points.push_back(Point2f(1.0,2.5));
      points.push_back(Point2f(2.0,23.5));
      points.push_back(Point2f(10.0,223.5));
      points.push_back(Point2f(1.5,24.5));

      // how these detected points map the new points at t + 1
      new_points.push_back(Point2f(1.1,2.5));
      new_points.push_back(Point2f(2.1,23.5));
      new_points.push_back(Point2f(10.1,223.5));
      new_points.push_back(Point2f(1.7,24.5));
      old_indices_1_2.push_back(0);
      old_indices_1_2.push_back(1);
      old_indices_1_2.push_back(2);
      old_indices_1_2.push_back(3);

      // points detected at time t + 1
      // NOTE: the 2nd and 3rd points is the same as new_points
      //       this is because there is high-chance that
      //       points matched from previous time will get picked as the corner
      //       for this time frame as well
      points2.push_back(Point2f(4.0,2.5));
      points2.push_back(Point2f(2.1,23.5));
      points2.push_back(Point2f(10.1,223.5));
      points2.push_back(Point2f(12.5,24.5));

      // Current situations:
      /* (1.1, 2.5)
         (2.1, 23.5)
         (10.1, 223.5)
         (1.7, 24.5)
         (4.0, 2.5)
         (1.7, 24.5)

         Connections: 1 --- 3
       */

      // how these detected points map to new points at time t + 2
      new_points2.push_back(Point2f(2.0, 15.5));
      new_points2.push_back(Point2f(1.7, 24.6));
      old_indices_2_3.push_back(1);
      old_indices_2_3.push_back(3);
  }

  // virtual void TearDown() {}

  ss::TrackManager track_manager_;
  vector<Point2f> points;
  vector<Point2f> new_points, new_points2;
  vector<Point2f> points2;
  vector<int> old_indices_1_2, // indices mapping from 1 to 2
            old_indices_2_3; // indices mapping from 2 to 3
};

TEST_F(SSTrackManagerTest, AddPoints){
    track_manager_.AddPoints(points);

    ASSERT_EQ(4, track_manager_.num_tracks());

    for (int i=0; i<track_manager_.num_tracks(); ++i){
        ASSERT_EQ(1, track_manager_.tracks()[i].number_of_times_tracked);
        ASSERT_FALSE(track_manager_.tracks()[i].activated);
    }
}

TEST_F(SSTrackManagerTest, AddPointsAndUpdate){
    track_manager_.AddPoints(points);
    ASSERT_EQ(4, track_manager_.num_tracks());

    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    ASSERT_EQ(4, track_manager_.num_tracks());
    ASSERT_NEAR(1.1, track_manager_.tracks()[0].pos.x, 0.001);
    ASSERT_NEAR(23.5, track_manager_.tracks()[1].pos.y, 0.001);
    ASSERT_NEAR(10.1, track_manager_.tracks()[2].pos.x, 0.001);
    ASSERT_NEAR(1.7, track_manager_.tracks()[3].pos.x, 0.001);

    // At time t+1, we detect a couple of points which happen to be the same as
    // points already in tracks
    track_manager_.AddPossiblyDuplicatePoints(points2);
    ASSERT_EQ(6, track_manager_.num_tracks());
    ASSERT_NEAR(12.5, track_manager_.tracks()[5].pos.x, 0.001);
    ASSERT_NEAR(223.5, track_manager_.tracks()[2].pos.y, 0.001);

    // these new points should not have been activated
    for (int i=4; i<track_manager_.num_tracks(); i++){
        ASSERT_FALSE(track_manager_.tracks()[i].activated);
    }
}

TEST_F(SSTrackManagerTest, RemoveDuplicatePoints){
    track_manager_.AddPoints(points);
    std::vector<cv::Point2f> new_points = points;
    track_manager_.RemoveDuplicatePoints(new_points);
    ASSERT_EQ(0, new_points.size());
}

TEST_F(SSTrackManagerTest, RemoveDuplicatePointsAfterUpdate){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    track_manager_.RemoveDuplicatePoints(points2);

    ASSERT_EQ(2, points2.size());
}

TEST_F(SSTrackManagerTest, TracksWhichPersistsLongEnoughGetActivated){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);

    // these points should have been activated
    for (int i=0; i<track_manager_.num_tracks(); i++){
        ASSERT_TRUE(track_manager_.tracks()[i].activated);
    }

    // Since these points are activated, they should now be connected to all the nearby nodes
    // NOTE: only track 1 and track 3 are close enough to each other.
    ASSERT_EQ(4, track_manager_.num_tracks());
    ASSERT_EQ(1, track_manager_.num_connections());
}

TEST_F(SSTrackManagerTest, MaxDistanceMinDistanceUpdate){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    track_manager_.AddPossiblyDuplicatePoints(points2);

    // NOTE: there is only 1 edge betwen track1 and track3
    // SEE: Test SSTrackManagerTest.TracksWhichPersistsLongEnoughGetActivated
    ss::LinkInformation edge_info_old, edge_info_new;
    track_manager_.get_edge_information(1,3, &edge_info_old);

    track_manager_.UpdatePoints(new_points2, old_indices_2_3);
    track_manager_.get_edge_information(1,3, &edge_info_new);

    // Since the points are moving away from each other, min_distance doesn't change
    // but max_distacen definitely should
    ASSERT_EQ(edge_info_old.id, edge_info_new.id);
    ASSERT_EQ(edge_info_old.min_distance, edge_info_new.min_distance);
    ASSERT_LT(edge_info_old.min_distance, edge_info_new.max_distance);
}


TEST_F(SSTrackManagerTest, GetEdgeInformation){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);

    ss::LinkInformation edge_info1;

    ASSERT_TRUE(track_manager_.get_edge_information(1,3, &edge_info1));
    ASSERT_NE(0, edge_info1.min_distance);
    ASSERT_NE(0, edge_info1.max_distance);

    // There are no other tracks
    ASSERT_FALSE(track_manager_.get_edge_information(1,2, &edge_info1));
    ASSERT_FALSE(track_manager_.get_edge_information(2,3, &edge_info1));
    ASSERT_FALSE(track_manager_.get_edge_information(0,3, &edge_info1));
    ASSERT_FALSE(track_manager_.get_edge_information(0,1, &edge_info1));
}
