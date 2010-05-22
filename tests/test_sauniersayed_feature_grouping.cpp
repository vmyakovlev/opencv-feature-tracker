#include <gtest/gtest.h>
#include <vector>
#include <cv.h>
#include "SaunierSayed_feature_grouping.h"

using std::vector;
using namespace cv;

namespace ss = SaunierSayed;

class SSTrackManagerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
      // points detected at time t
      points.push_back(Point2f(1.0,2.5));
      points.push_back(Point2f(2.0,23.5));
      points.push_back(Point2f(10.0,223.5));
      points.push_back(Point2f(1.5,24.5));

      // how these detected points map the new points
      new_points.push_back(Point2f(1.1,2.5));
      new_points.push_back(Point2f(2.1,23.5));
      new_points.push_back(Point2f(10.1,223.5));
      new_points.push_back(Point2f(1.7,24.5));
      old_indices_1_2.push_back(0);
      old_indices_1_2.push_back(1);
      old_indices_1_2.push_back(2);
      old_indices_1_2.push_back(3);

      // points detected at time t + 1
      points2.push_back(Point2f(2.0,2.5));
      points2.push_back(Point2f(3.0,2.5));
      points2.push_back(Point2f(12.0,23.5));
      points2.push_back(Point2f(12.5,24.5));
  }

  // virtual void TearDown() {}

  ss::TrackManager track_manager_;
  vector<Point2f> points;
  vector<Point2f> new_points;
  vector<Point2f> points2;
  vector<int> old_indices_1_2; // indices mapping from 1 to 2
};

TEST_F(SSTrackManagerTest, AddPoints){
    track_manager_.AddPoints(points);

    ASSERT_EQ(4, track_manager_.tracks().size());
}

TEST_F(SSTrackManagerTest, AddPointsAndUpdate){
    track_manager_.AddPoints(points);
    ASSERT_EQ(4, track_manager_.tracks().size());

    track_manager_.UpdatePoints(new_points, old_indices_1_2);
    ASSERT_EQ(4, track_manager_.tracks().size());
    ASSERT_NEAR(1.1, track_manager_.tracks()[0].pos.x, 0.001);
    ASSERT_NEAR(23.5, track_manager_.tracks()[1].pos.y, 0.001);

}
