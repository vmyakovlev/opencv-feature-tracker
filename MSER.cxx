#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include "emd_matcher.h"
#include "misc.h"
#include "draw.h"
#include "Blob.h"
#include "blob_feature.h"
#include "window_pair.h"

using std::string;
using std::vector;
using namespace cv;

/** \file MSER.cxx Compute MSER from image for matching purposes
*/
int main(int argc, char ** argv){
    if (argc<3){
        printf("Usage: %s [options] input_image.jpg input_image2.jpg",argv[0]);
        exit(-1);
    }	

    // MSER object
    MSER mser = MSER();

    // Load image
    Mat im = imread(argv[1]);

    // make a copy for visualization
    Mat im2 = imread(argv[2]);

    // Compute MSER
    vector<vector<Point> > im_mser_points;
    vector<vector<Point> > im2_mser_points;

    mser(im, im_mser_points, Mat());
    mser(im2, im2_mser_points, Mat());

    // Convert MSER to blobs
    vector<Blob> im_blobs;
    vector<Blob> im2_blobs;

    // NOTE: I am sure Boost has this implemented in a much nicer way in their library
    //       I simply do not want my teammate having to install Boost on their system just for this
    vector_one_to_another<vector<Point>,Blob>(im_mser_points, im_blobs);
    vector_one_to_another<vector<Point>,Blob>(im2_mser_points, im2_blobs);

    // Compute EMD between blobs
    std::cout << "Compute EMD" << std::endl;

    vector<KeyPoint> im_keypoints, im2_keypoints;
    vector_one_to_another<Blob,KeyPoint>(im_blobs, im_keypoints);
    vector_one_to_another<Blob,KeyPoint>(im2_blobs, im2_keypoints);

    Mat im_descriptors, im2_descriptors;

    BlobDescriptorExtractor descriptor_extractor;
    descriptor_extractor.compute(im,
                                 im_keypoints,
                                 im_descriptors);
    descriptor_extractor.compute(im2,
                                 im2_keypoints,
                                 im2_descriptors);
    // get weights arrays
    // we are weighting the descriptor by the detected blob size
    Mat im_weight(im_descriptors.rows, 1, CV_32F);
    Mat im2_weight(im2_descriptors.rows, 1, CV_32F);
    for (int i=0; i<im_descriptors.rows; i++){
        im_weight.at<float>(i,0) = im_blobs[i].area();
    }
    for (int i=0; i<im2_descriptors.rows; i++){
        im2_weight.at<float>(i,0) = im2_blobs[i].area();
    }

    // find matching blobs
    EMDDescriptorMatcher emd_matcher;
    IndexPairs matching_indexes;
    vector<float> matching_strength;

    emd_matcher.SetWeights(im_weight, im2_weight);
    emd_matcher.match(im_descriptors,im2_descriptors, matching_indexes, matching_strength);

    ///////////////////////////////////////////////////////////
    // VISUALIZATION

    // Draw polylines of contours on images
    // polylines(im, im_mser_points, true, CV_RGB(50,50,0));

    // Draw bounding box and center of each blobs
    // Also, draw enclosing circles
    for (int i=0; i<im_blobs.size(); i++){
        im_blobs[i].draw_to(im);
    }

    for (int i=0; i<im2_blobs.size(); i++){
        im2_blobs[i].draw_to(im2);
    }

    string window = "MSER";
    WindowPair window_pair(im,im2,window);

    // Draw arrows for the matching
    Point from,to;
    for( int i=0; i<matching_indexes.size(); i++){
        from = im_keypoints[matching_indexes[i].first].pt;
        to = im2_keypoints[matching_indexes[i].second].pt;

        window_pair.DrawArrow(from,to, Scalar(255,255,0));
    }

    // Display
    std::cout << "Displaying Window" << std::endl;
    window_pair.Show();

    // Save output matching
    window_pair.Save("pair_matching.png");

    return 0;
}
