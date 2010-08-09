#include "misc.h"
#include "test_configuration.h"
#include <gtest/gtest.h>

#include <cv.h>
#include <highgui.h>

using namespace cv;

TEST(TestMisc, Indexing){
    vector<int> points;
    points.push_back(4);
    points.push_back(1);
    points.push_back(2);
    points.push_back(5);
    points.push_back(3);

    vector<int> new_indices;
    new_indices.push_back(2);
    new_indices.push_back(3);
    new_indices.push_back(4);
    new_indices.push_back(1);
    new_indices.push_back(0);

    vector<int> indexed_points = indexing(points, new_indices);

    ASSERT_EQ(2, indexed_points[0]);
    ASSERT_EQ(5, indexed_points[1]);
    ASSERT_EQ(3, indexed_points[2]);
    ASSERT_EQ(1, indexed_points[3]);
    ASSERT_EQ(4, indexed_points[4]);
}

TEST(TestMisc, UnWarpUsingHomographyFromPoints){
    Mat corresponding_pairs = loadtxt(data_folder_path + "/test1.avi.homography.txt");

    Mat image_points = corresponding_pairs.rowRange(0,4);
    printf("image_points =\n");
    print_matrix<float>(image_points);

    Mat world_points = corresponding_pairs.rowRange(4,8);
    printf("world_points =\n");
    print_matrix<float>(world_points);

    Mat homography_matrix = findHomography(image_points, world_points);

    ASSERT_EQ(3, homography_matrix.cols);
    ASSERT_EQ(3, homography_matrix.rows);

    printf("homography_matrix =\n");
    print_matrix<float>(homography_matrix);

    // unwrap image
    Mat im = cv::imread(data_folder_path + "test_avi1.png");
    Mat unwarped_image;
//    cv::warpPerspective(im, unwarped_image, homography_matrix, cv::Size(400,400));
//    cv::imwrite("unwarped_frame_from_poins.png", unwarped_image);
}

TEST(TestMisc, UnWarpUsingHomographyMatrixImage){
    Mat homography_matrix = loadtxt(data_folder_path + "/test2.avi.homography.mat");

    ASSERT_EQ(3, homography_matrix.cols);
    ASSERT_EQ(3, homography_matrix.rows);

    Mat image = cv::imread(data_folder_path + "/test_frame_avi2.png", 0);

    ASSERT_EQ(1, image.channels());

    // Try to unwarp the image
    Mat unwarped_image;
    cv::warpPerspective(image, unwarped_image, homography_matrix, cv::Size(256,256), CV_WARP_INVERSE_MAP);

    cv::imwrite("unwarped_frame_from_file.png", unwarped_image);
}

TEST(TestMisc, UnWarpUsingHomographyMatrix){
    Mat homography_matrix = loadtxt(data_folder_path + "/test2.avi.homography.mat");

    int num_points = 6;
    float a[] = {1,1, 100, 101, 150, 151, 170, 171, 50,0, 50, 50};
    Mat a_mat(num_points,1,CV_32FC2, a);

    Mat unwarped_a;
    Mat invert_homography_matrix;
    cv::invert(homography_matrix, invert_homography_matrix);
    cv::perspectiveTransform(a_mat, unwarped_a, invert_homography_matrix);

    ASSERT_EQ(num_points, unwarped_a.rows);
    ASSERT_EQ(2, unwarped_a.channels());

    printf("a =\n");
    print_matrix<cv::Vec2f>(a_mat);

    printf("unwarped_a =\n");
    print_matrix<cv::Vec2f>(unwarped_a);
}

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

TEST(TestMisc, ConvertPointsFromImageToWorld){
    /* In order to get these data, do this in ipython -pylab
    points = np.array([[1,3,2,3],[2,3,4,5],[1,1,1,1]], dtype=np.float)
    homography = np.array([[1,2,3],[2,3,1],[2,2,1]])
    proj_points = np.dot(homography,points)
    proj_points_2 = proj_points / proj_points[2,:]

    In [18]: proj_points_2
    Out[18]:
    array([[ 1.14285714,  0.92307692,  1.        ,  0.94117647],
           [ 1.28571429,  1.23076923,  1.30769231,  1.29411765],
           [ 1.        ,  1.        ,  1.        ,  1.        ]])
    */

    // Create the points
    std::vector<Point2f> points;

    points.push_back(cv::Point2f(1,2));
    points.push_back(cv::Point2f(3,3));
    points.push_back(cv::Point2f(2,4));
    points.push_back(cv::Point2f(3,5));

    // Create the homography matrix
    Mat homography_matrix(3,3, CV_32F);
    homography_matrix.at<float>(0,0) = 1;
    homography_matrix.at<float>(0,1) = 2;
    homography_matrix.at<float>(0,2) = 3;
    homography_matrix.at<float>(1,0) = 2;
    homography_matrix.at<float>(1,1) = 3;
    homography_matrix.at<float>(1,2) = 1;
    homography_matrix.at<float>(2,0) = 2;
    homography_matrix.at<float>(2,1) = 2;
    homography_matrix.at<float>(2,2) = 1;

    // Get the projected_points
    std::vector<Point2f> proj_points;
    convert_to_world_coordinate(points, homography_matrix, &proj_points);

    ASSERT_EQ(4, proj_points.size());

    // Check the projected points
    ASSERT_NEAR(1.14285714, proj_points[0].x, 0.00001);
    ASSERT_NEAR(0.92307692, proj_points[1].x, 0.00001);
    ASSERT_NEAR(1., proj_points[2].x, 0.00001);
    ASSERT_NEAR(0.94117647, proj_points[3].x, 0.00001);

    ASSERT_NEAR(1.28571429, proj_points[0].y, 0.00001);
    ASSERT_NEAR(1.23076923, proj_points[1].y, 0.00001);
    ASSERT_NEAR(1.30769231, proj_points[2].y, 0.00001);
    ASSERT_NEAR(1.29411765, proj_points[3].y, 0.00001);
}

TEST(TestMisc, ConvertPointsFromWorldToImage){
    // See Test_ConvertPointsFromImageToWorld for the data

    // Create the points
    std::vector<Point2f> points;

    points.push_back(cv::Point2f(1.14285714,1.28571429));
    points.push_back(cv::Point2f(0.92307692,1.23076923));
    points.push_back(cv::Point2f(1,1.30769231));
    points.push_back(cv::Point2f(0.94117647,1.29411765));

    // Create the homography matrix
    Mat homography_matrix(3,3, CV_32F);
    homography_matrix.at<float>(0,0) = 1;
    homography_matrix.at<float>(0,1) = 2;
    homography_matrix.at<float>(0,2) = 3;
    homography_matrix.at<float>(1,0) = 2;
    homography_matrix.at<float>(1,1) = 3;
    homography_matrix.at<float>(1,2) = 1;
    homography_matrix.at<float>(2,0) = 2;
    homography_matrix.at<float>(2,1) = 2;
    homography_matrix.at<float>(2,2) = 1;

    // Get the projected_points
    std::vector<Point2f> proj_points;
    convert_to_image_coordinate(points, homography_matrix, &proj_points);

    ASSERT_EQ(4, proj_points.size());

    // Check the projected points
    ASSERT_NEAR(1, proj_points[0].x, 0.00001);
    ASSERT_NEAR(3, proj_points[1].x, 0.00001);
    ASSERT_NEAR(2., proj_points[2].x, 0.00001);
    ASSERT_NEAR(3, proj_points[3].x, 0.00001);

    ASSERT_NEAR(2, proj_points[0].y, 0.00001);
    ASSERT_NEAR(3, proj_points[1].y, 0.00001);
    ASSERT_NEAR(4, proj_points[2].y, 0.00001);
    ASSERT_NEAR(5, proj_points[3].y, 0.00001);
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

TEST(TestMisc, ConstReferencePushBack){
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(2.5,1.6));
    points.push_back(cv::Point2f(25,1.3));
    points.push_back(cv::Point2f(2.6,1.2));
    points.push_back(cv::Point2f(27,3.6));

    int the_size = points.size();

    points.push_back(points.at(points.size() - 1));

    ASSERT_EQ(the_size+1, points.size());
    ASSERT_EQ(points[3].x, points[4].x);
    ASSERT_EQ(points[3].y, points[4].y);
}
