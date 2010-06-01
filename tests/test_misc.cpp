#include "misc.h"
#include "test_configuration.h"
#include <gtest/gtest.h>

#include <cv.h>
using cv::Mat;
using cv::KeyPoint;
using cv::Point2f;

TEST(TestMisc, Dataset1){
    Mat loaded_mat = loadtxt(data_folder_path + "/query_points1.txt");

    ASSERT_EQ(2, loaded_mat.cols);
    ASSERT_EQ(8427, loaded_mat.rows);
}

TEST(TestMisc, Dataset2){
    Mat loaded_mat = loadtxt(data_folder_path + "/query_points2.txt");

    ASSERT_EQ(2, loaded_mat.cols);
    ASSERT_EQ(14256, loaded_mat.rows);
}

TEST(TestMisc, BadFileName){
    ASSERT_THROW(
        Mat loaded_mat = loadtxt("DOESNOTEXISTS.BLAHBLAH"),
        cv::Exception
        );
}

TEST(TestMisc, ConvertKeyPointToPoint2fAndBack){
    std::vector<KeyPoint> keypoints;
    keypoints.push_back( KeyPoint(Point2f(1,2),1) );
    keypoints.push_back( KeyPoint(Point2f(2,3),1) );
    keypoints.push_back( KeyPoint(Point2f(3,4),1) );
    keypoints.push_back( KeyPoint(Point2f(4,5),1) );

    std::vector<Point2f> points;
    vector_one_to_another(keypoints, points);

    ASSERT_EQ(1, points[0].x);
    ASSERT_EQ(2, points[0].y);
    ASSERT_EQ(2, points[1].x);
    ASSERT_EQ(3, points[1].y);
    ASSERT_EQ(3, points[2].x);
    ASSERT_EQ(4, points[2].y);
    ASSERT_EQ(4, points[3].x);
    ASSERT_EQ(5, points[3].y);

    points.pop_back();
    vector_one_to_another(points, keypoints);

    ASSERT_EQ(3, keypoints.size());
    ASSERT_EQ(1, keypoints[0].pt.x);
    ASSERT_EQ(2, keypoints[0].pt.y);
    ASSERT_EQ(2, keypoints[1].pt.x);
    ASSERT_EQ(3, keypoints[1].pt.y);
    ASSERT_EQ(3, keypoints[2].pt.x);
    ASSERT_EQ(4, keypoints[2].pt.y);
}

TEST(TestOpenCV, SimpleColumnMultiplyAdd){
    Mat column_a = Mat::ones(4,1,CV_32F);
    Mat column_b = Mat::ones(4,1,CV_32F);
    Mat column_c = column_a * 200 + column_b;

    Mat column_d;
    column_c.convertTo(column_d, CV_32S);

    ASSERT_EQ(201, column_c.at<float>(0,0));
    ASSERT_EQ(201, column_c.at<float>(1,0));
    ASSERT_EQ(201, column_c.at<float>(2,0));
    ASSERT_EQ(201, column_c.at<float>(3,0));

    ASSERT_EQ(201, column_d.at<int>(0,0));
    ASSERT_EQ(201, column_d.at<int>(1,0));
    ASSERT_EQ(201, column_d.at<int>(2,0));
    ASSERT_EQ(201, column_d.at<int>(3,0));
}

TEST(TestOpenCV, CreateEmptyMatrix){
    ASSERT_TRUE(Mat().empty());
}

TEST(TestOpenCV, ConvertVectorPoint2fToMat){
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(2.5,1.6));
    points.push_back(cv::Point2f(25,1.3));
    points.push_back(cv::Point2f(2.6,1.2));
    points.push_back(cv::Point2f(27,3.6));

    // This won't work
    // Mat points_mat = points;

    // Convert points (traditional conversion)
    Mat points_mat(points.size(), 1, CV_32FC2);
    points_mat.at< cv::Vec2f >(0,0) = points[0];
    points_mat.at< cv::Vec2f >(1,0) = points[1];
    points_mat.at< cv::Vec2f >(2,0) = points[2];
    points_mat.at< cv::Vec2f >(3,0) = points[3];

    ASSERT_NEAR(2.5, points_mat.at<cv::Vec2f>(0,0)[0], 0.001);
    ASSERT_NEAR(1.3, points_mat.at<cv::Vec2f>(1,0)[1], 0.001);
    ASSERT_NEAR(25, points_mat.at<cv::Vec2f>(1,0)[0], 0.001);

    // Convert points (use explicit construction from vector)
    Mat points_mat2(points, false);
    ASSERT_NEAR(2.5, points_mat2.at<cv::Vec2f>(0,0)[0], 0.001);
    ASSERT_NEAR(1.3, points_mat2.at<cv::Vec2f>(1,0)[1], 0.001);
    ASSERT_NEAR(25, points_mat2.at<cv::Vec2f>(1,0)[0], 0.001);
    ASSERT_EQ(4, points_mat2.rows);
    ASSERT_EQ(1, points_mat2.cols);
    ASSERT_EQ(2, points_mat2.channels());
}
