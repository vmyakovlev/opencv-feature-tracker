#include <cv.h>
#include <highgui.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>

#include "misc.h"
#include "draw.h"
#include "window_pair.h"

using std::string;
using std::vector;
using namespace cv;

/** \brief Simple conversion until surf() got adapted to the new framework */
void surf_descriptor_vec_to_mat(const vector<float> vec, Mat & mat){
    int vec_size = vec.size();

    mat = Mat(vec_size/128,128,CV_32F);
    float * mat_ptr = mat.ptr<float>(0);
    for (int i=0; i<vec_size; i++){
        *mat_ptr = vec[i];
        mat_ptr++;
    }
}

/** \file SURF.cxx Compute SURF feature and match them
*/
int main(int argc, char ** argv){
    if (argc<3){
        printf("Usage: %s [options] input_image.jpg input_image2.jpg",argv[0]);
        exit(-1);
    }

    // MSER object
    SURF surf = SURF(0.5);

    // Load image
    Mat im_in = imread(argv[1]);
    Mat im;
    cvtColor(im_in, im, CV_RGB2GRAY);

    // make a copy for visualization
    Mat im2_in = imread(argv[2]);
    Mat im2;
    cvtColor(im2_in, im2, CV_RGB2GRAY);

    // Compute SURF
    vector<KeyPoint> im_keypoints, im2_keypoints;
    vector<float> im_descriptors_vec, im2_descriptors_vec;
    surf(im, Mat(), im_keypoints, im_descriptors_vec);
    surf(im2, Mat(), im2_keypoints, im2_descriptors_vec);

    std::cout << "Keypoints " << im_keypoints.size() << " " << im2_keypoints.size() << std::endl
            << "Descriptors " << im_descriptors_vec.size()*1.0/im_keypoints.size() << " " << im2_descriptors_vec.size()*1.0/im2_keypoints.size() << std::endl;

    Mat im_descriptors, im2_descriptors;
    surf_descriptor_vec_to_mat(im_descriptors_vec, im_descriptors);
    surf_descriptor_vec_to_mat(im2_descriptors_vec, im2_descriptors);

    // construct ANN search tree
    std::cout << "Construct ANN search tree" << std::endl;
    cv::flann::KDTreeIndexParams search_tree_params(4);
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

    string window = "SURF";
    WindowPair window_pair(im_in,im2_in,window);

    // Draw arrows for the matching
    Point2f from,to;
    for( int i=0; i<resulting_indices.rows; i++){
        from = im_keypoints[i].pt;
        to = im2_keypoints[resulting_indices.at<int>(i,0)].pt;

        window_pair.DrawArrow(from,to, Scalar(255,255,0));
        window_pair.Show();
    }

    // Display
    std::cout << "Displaying Window" << std::endl;


    std::cout << "Done SURF-ing"<< std::endl;

    return 0;
}
