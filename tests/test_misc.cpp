#include "misc.h"
#include "test_configuration.h"
#include <gtest/gtest.h>

#include <cv.h>
using cv::Mat;

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
