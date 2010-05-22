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
      //track_manager_ = ss::TrackManager();
  }

  // virtual void TearDown() {}

  ss::TrackManager track_manager_;
};

TEST_F(SSTrackManagerTest, AddPoints){
    // some new points
    vector<Point2f> points;
    points.push_back(Point2f(1.0,2.5));
    points.push_back(Point2f(2.0,23.5));
    points.push_back(Point2f(10.0,223.5));
    points.push_back(Point2f(1.5,24.5));

    track_manager_.AddPoints(points);

    ASSERT_EQ(4, track_manager_.tracks.size());
}
