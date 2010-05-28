#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include "misc.h"
#include "draw.h"
#include "daisy_feature.h"
#include "window_pair.h"

using std::string;
using std::vector;
using namespace cv;

/** \file DAISY.cxx Compute DAISY from image for matching purposes
  \todo Fix the problem of Daisy complaining about destroyed memory on its dtor
*/
int main(int argc, char ** argv){
    if (argc<3){
        printf("Usage: %s [options] input_image.jpg input_image2.jpg",argv[0]);
        exit(-1);
    }

    // Load image
    Mat im = imread(argv[1],0);

    // make a copy for visualization
    Mat im2 = imread(argv[2],0);

    string window = "DAISY";
    WindowPair window_pair(im,im2,window);

    vector<Point2f> im_corners, im2_corners;
    // find good features to track (for now, we will use Harris corner)
    goodFeaturesToTrack(im, im_corners, 50, 0.20, 1, Mat());
    goodFeaturesToTrack(im2, im2_corners, 50, 0.20, 1, Mat());

    // Compute DAISY keypoints
    std::cout << "Compute DAISY for keypoints" << std::endl;

    vector<KeyPoint> im_keypoints, im2_keypoints;
    vector_one_to_another<Point2f,KeyPoint>(im_corners, im_keypoints);
    vector_one_to_another<Point2f,KeyPoint>(im2_corners, im2_keypoints);

    Mat im_descriptors, im2_descriptors;

    DaisyDescriptorExtractor descriptor_extractor;

    std::cout << "Compute DAISY descriptor for first image" << std::endl;
    descriptor_extractor.compute(im,
                                 im_keypoints,
                                 im_descriptors);
    std::cout << "Compute DAISY descriptor for second image" << std::endl;
    descriptor_extractor.compute(im2,
                                 im2_keypoints,
                                 im2_descriptors);

    // construct ANN search tree
    std::cout << "Construct ANN search tree" << std::endl;
    //cv::flann::KDTreeIndexParams search_tree_params(4);
    cv::flann::LinearIndexParams search_tree_params;
    cv::flann::Index search_tree(im_descriptors, search_tree_params);

    // perform query for descriptors in the 2nd image
    std::cout << "Perform query for descriptors in the 2nd image" << std::endl;
    static int num_knn_to_search = 5;
    cv::flann::SearchParams search_params;
    Mat resulting_indices(im2_descriptors.rows, num_knn_to_search, CV_32S);
    Mat resulting_distances(im2_descriptors.rows, num_knn_to_search, CV_32F);
    search_tree.knnSearch(im2_descriptors, resulting_indices, resulting_distances, num_knn_to_search, search_params);

    // visualizing results
    std::cout << "Visualizing DAISY results" << std::endl;

    // Draw arrows for the matching
    Point2f from,to;
    for( int i=0; i<resulting_indices.rows; i++){
        from = im_corners[i];
        to = im2_corners[resulting_indices.at<int>(i,0)];

        window_pair.DrawArrow(from,to, Scalar(255,255,0));
    }

    // Display
    std::cout << "Displaying Window" << std::endl;
    window_pair.Show();

    std::cout << "Done DAISY-ing"<< std::endl;

    return 0;
}
