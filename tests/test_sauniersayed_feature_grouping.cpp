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
    // NOTE: this constructor with initialization list of track_manager_ is required
    //       because for some reason track_manager_ will not get the required parameters
    SSTrackManagerTest():
        track_manager_(2, 0.05, 4, 20, 0, 0.05)
    {

    }
  virtual void SetUp() {
      // points detected at time t
      points.push_back(Point2f(1.0,2.5));
      points.push_back(Point2f(2.0,23.5));
      points.push_back(Point2f(10.0,223.5));
      points.push_back(Point2f(1.5,24.5));
      points.push_back(Point2f(200,100));

      // how these detected points map the new points at t + 1
      new_points.push_back(Point2f(1.2,2.5));
      new_points.push_back(Point2f(2.2,23.5));
      new_points.push_back(Point2f(10.2,223.5));
      new_points.push_back(Point2f(1.7,24.5));
      new_points.push_back(Point2f(200.04,100));
      old_indices_1_2.push_back(0);
      old_indices_1_2.push_back(1);
      old_indices_1_2.push_back(2);
      old_indices_1_2.push_back(3);
      old_indices_1_2.push_back(4);

      // points detected at time t + 1
      // NOTE: the 2nd and 3rd points is the same as new_points
      //       this is because there is high-chance that
      //       points matched from previous time will get picked as the corner
      //       for this time frame as well
      points2.push_back(Point2f(4.0,2.5));
      points2.push_back(Point2f(2.2,23.5));
      points2.push_back(Point2f(10.2,223.5));
      points2.push_back(Point2f(12.5,24.5));

      // Current situations:
      /* (1.2, 2.5)
         (2.2, 23.5)
         (10.2, 223.5)
         (1.7, 24.5)
         (200.04, 100)
         (4.0, 2.5)
         (12.5, 24.5)

         Connections: 1 --- 3
       */

      // how these detected points map to new points at time t + 2
      new_points2.push_back(Point2f(2.0, 15.5));
      new_points2.push_back(Point2f(1.7, 24.6));
      new_points2.push_back(Point2f(200, 100));
      old_indices_2_3.push_back(1);
      old_indices_2_3.push_back(3);
      old_indices_2_3.push_back(4);

      // NOTE: we design this update so that the point (200,100) fluctuates a little bit
      //       but not actually go anywhere. This point is a point on a static object in
      //       the scene. Hence, it should not be activated.

      // how these detected points map to new points at time t + 3
      new_points3.push_back(Point2f(1, 0));
      new_points3.push_back(Point2f(1.7, 54.6));
      old_indices_3_4.push_back(1);
      old_indices_3_4.push_back(3);
  }

  // virtual void TearDown() {}

  ss::TrackManager track_manager_;
  vector<Point2f> points;
  vector<Point2f> new_points, new_points2, new_points3;
  vector<Point2f> points2;
  vector<int> old_indices_1_2, // indices mapping from 1 to 2
            old_indices_2_3, // indices mapping from 2 to 3
            old_indices_3_4;
};

TEST_F(SSTrackManagerTest, AddPoints){
    std::vector<int> assigned_ids;
    track_manager_.AddPoints(points, &assigned_ids);

    ASSERT_EQ(5, track_manager_.num_tracks());

    // All points are connected to all other tracks but they are all inactive
    ASSERT_EQ(10, track_manager_.num_connections());

    // Check that these connections are inactive
    ss::LinkInformation link_info;
    for (int i=0; i<track_manager_.num_tracks(); ++i){
        for (int j=i+1; j<track_manager_.num_tracks(); ++j){
            track_manager_.get_edge_information(i, j, &link_info);
            ASSERT_FALSE(link_info.active);
        }
    }

    for (int i=0; i<track_manager_.num_tracks(); ++i){
        ASSERT_EQ(i, assigned_ids[i]);
        ASSERT_EQ(1, track_manager_.tracks()[i].number_of_times_tracked);
        ASSERT_FALSE(track_manager_.tracks()[i].activated);
    }
}

TEST_F(SSTrackManagerTest, GetAllTrackPositions){
    std::vector<int> assigned_ids;
    track_manager_.AddPoints(points, &assigned_ids);

    vector<Point2f> positions;
    track_manager_.GetAllTracksPositionAndId(&positions, &assigned_ids);

    ASSERT_EQ(5, positions.size());

    for (int i=0; i<track_manager_.num_tracks(); ++i){
        ASSERT_EQ(points[i].x, positions[i].x);
        ASSERT_EQ(points[i].y, positions[i].y);
        ASSERT_EQ(i, assigned_ids[i]);
    }
}

TEST_F(SSTrackManagerTest, AddPointsAndUpdate){
    track_manager_.AddPoints(points);
    ASSERT_EQ(5, track_manager_.num_tracks());

    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    ASSERT_EQ(5, track_manager_.num_tracks());

    ss::Tracks tracks = track_manager_.tracks();

    ASSERT_NEAR(1.2, tracks[0].pos.x, 0.001);
    ASSERT_NEAR(23.5, tracks[1].pos.y, 0.001);
    ASSERT_NEAR(10.2, tracks[2].pos.x, 0.001);
    ASSERT_NEAR(1.7, tracks[3].pos.x, 0.001);

    // At time t+1, we detect a couple of points which happen to be the same as
    // points already in tracks
    std::vector<int> assigned_ids;
    track_manager_.AddPossiblyDuplicatePoints(points2, &assigned_ids);

    tracks = track_manager_.tracks();
    ASSERT_EQ(7, track_manager_.num_tracks());
    ASSERT_NEAR(12.5, tracks[6].pos.x, 0.001);
    ASSERT_NEAR(223.5, tracks[2].pos.y, 0.001);

    // check that the displacements are recorded for updated point
    ASSERT_EQ(1, tracks[2].previous_displacements.size());
    // but displacements are not recorded for newly added point
    ASSERT_EQ(0, tracks[6].previous_displacements.size());

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

TEST_F(SSTrackManagerTest, FindDuplicatePoints){
    track_manager_.AddPoints(points);

    std::vector<int> assigned_ids;
    track_manager_.FindDuplicatePointIds(points, &assigned_ids);

    ASSERT_EQ(0, assigned_ids[0]);
    ASSERT_EQ(1, assigned_ids[1]);
    ASSERT_EQ(2, assigned_ids[2]);
    ASSERT_EQ(1, assigned_ids[3]); //NOTE: point[1] and point[3] are too close together
}

TEST_F(SSTrackManagerTest, RemoveDuplicatePointsAfterUpdate){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    track_manager_.RemoveDuplicatePoints(points2);

    ASSERT_EQ(2, points2.size());
}

TEST_F(SSTrackManagerTest, AddPossibleDuplicatePointsAndGetAssignedIds){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);

    std::vector<int> assigned_ids;
    track_manager_.FindDuplicatePointIds(points2, &assigned_ids);

    ASSERT_EQ(-1, assigned_ids[0]);
    ASSERT_EQ(1, assigned_ids[1]);
    ASSERT_EQ(2, assigned_ids[2]);
    ASSERT_EQ(-1, assigned_ids[3]);

    track_manager_.AddPossiblyDuplicatePoints(points2, &assigned_ids);

    ASSERT_EQ(5, assigned_ids[0]); // not duplicate => new id
    ASSERT_EQ(1, assigned_ids[1]); // duplicate => old id
    ASSERT_EQ(2, assigned_ids[2]); // duplicate => old id
    ASSERT_EQ(6, assigned_ids[3]); // not duplicate => new id
}

TEST_F(SSTrackManagerTest, TracksWhichPersistsLongEnoughGetActivated){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    vector<int> assigned_ids;
    track_manager_.AddPossiblyDuplicatePoints(points2, &assigned_ids);

    // 0-3 points should have been activated
    ASSERT_TRUE(track_manager_.tracks()[0].activated);
    ASSERT_TRUE(track_manager_.tracks()[1].activated);
    ASSERT_TRUE(track_manager_.tracks()[2].activated);
    ASSERT_TRUE(track_manager_.tracks()[3].activated);

    // the last point should not be activated since it is a stationary point
    ASSERT_FALSE(track_manager_.tracks()[4].activated);

    // Since these points are activated, they should now be connected to all the nearby nodes
    // NOTE: only track 1 and track 3 are close enough to each other.
    ASSERT_EQ(7, track_manager_.num_tracks());
    ss::LinkInformation link_info;
    track_manager_.get_edge_information(1, 3, &link_info);
    ASSERT_TRUE(link_info.active);

    // Another time step, only 0-3 are updated
    track_manager_.UpdatePoints(new_points2, old_indices_2_3);
    ss::Tracks current_track_information = track_manager_.tracks();
    ASSERT_TRUE(current_track_information[0].activated);
    ASSERT_TRUE(current_track_information[1].activated);
    ASSERT_TRUE(current_track_information[2].activated);
    ASSERT_TRUE(current_track_information[3].activated);
    ASSERT_FALSE(current_track_information[4].activated);
    ASSERT_FALSE(current_track_information[5].activated);
    ASSERT_FALSE(current_track_information[6].activated);
}

TEST_F(SSTrackManagerTest, MaxDistanceMinDistanceUpdate){
    track_manager_.AddPoints(points);

    vector<int> assigned_ids;
    ss::LinkInformation edge_info_old, edge_info_new;

    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    ASSERT_TRUE(track_manager_.get_edge_information(1,3, &edge_info_old));

    track_manager_.AddPossiblyDuplicatePoints(points2, &assigned_ids);

    // NOTE: there is only 1 edge betwen track1 and track3
    // SEE: Test SSTrackManagerTest.TracksWhichPersistsLongEnoughGetActivated
    ASSERT_TRUE(track_manager_.get_edge_information(1,3, &edge_info_old));

    track_manager_.UpdatePoints(new_points2, old_indices_2_3);
    ASSERT_TRUE(track_manager_.get_edge_information(1,3, &edge_info_new));

    // Since the points are moving away from each other, min_distance doesn't change
    // but max_distace definitely should
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
}

TEST_F(SSTrackManagerTest, EdgeIsSeveredIfDistanceTooLarge){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    track_manager_.UpdatePoints(new_points2, old_indices_2_3);
    track_manager_.UpdatePoints(new_points3, old_indices_3_4);

    // NOTE: the last point update was meant to break any possible link between 1 and 3
    ss::LinkInformation edge_info;
    ASSERT_FALSE(track_manager_.get_edge_information(1,3, &edge_info));
}

TEST_F(SSTrackManagerTest, GetConnectedComponents){
    track_manager_.AddPoints(points);
    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    track_manager_.UpdatePoints(new_points2, old_indices_2_3);

    ss::ConnectedComponents connected_components = track_manager_.GetConnectedComponents();

    // There are 4 components: one that has 3 element, the other two have 1 elements
    ASSERT_EQ(4, connected_components.size());

    ASSERT_EQ(1, connected_components[0].size());

    ASSERT_EQ(1, connected_components[1][0].id);
    ASSERT_EQ(3, connected_components[1][1].id);

    ASSERT_EQ(1, connected_components[2].size());
}
