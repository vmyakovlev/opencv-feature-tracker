#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include "misc.h"
#include "draw.h"
#include "daisy_feature.h"

using std::string;
using std::vector;
using namespace cv;

/** \file InteractiveMatching.cxx Interactively match selected point in descriptor space
*/
int main(int argc, char ** argv){
    if (argc<4){
        printf("Usage: %s [options] input_image.jpg input_image2.jpg location.txt",argv[0]);
        exit(-1);
    }

    /////////////////////////////////////////////////////////
    // PARAMETERS
    string window1 = "Source image";
    string window2 = "Target image";
    const int num_knn_to_search = 5;

    /////////////////////////////////////////////////////////
    // Load image
    Mat gray_im = imread(argv[1],0);
    Mat im = imread(argv[1]);

    // make a copy for visualization
    Mat gray_im2 = imread(argv[2],0);
    Mat im2 = imread(argv[2]);

    // Compute DAISY
    std::cout << "Compute dense DAISY" << std::endl;
    DaisyDescriptorExtractor descriptor_extractor;

    Mat dense_descriptors_im,dense_descriptors_im2;

    std::cout << "Compute DAISY descriptor for first image" << std::endl;
    descriptor_extractor.compute_dense(gray_im, dense_descriptors_im);
    std::cout << "Compute DAISY descriptor for second image" << std::endl;
    descriptor_extractor.compute_dense(gray_im2, dense_descriptors_im2);

    // construct ANN search tree
    std::cout << "Construct ANN search tree" << std::endl;
    cv::flann::KDTreeIndexParams search_tree_params(4);
    //cv::flann::LinearIndexParams search_tree_params;
    cv::flann::Index * search_tree = new cv::flann::Index(dense_descriptors_im, search_tree_params);

    // DEBUG: saving stuff for debugging analysis
    //std::cout << "Saving features for debug purposes" << std::endl;
    //cvSave("/home/dchu/Desktop/im1.cvmat", &CvMat(dense_descriptors_im));
    //cvSave("/home/dchu/Desktop/im2.cvmat", &CvMat(dense_descriptors_im2));

    // ready window for interactive display of matches
    std::cout << "Visualizing DAISY results" << std::endl;

    int descriptor_length = dense_descriptors_im.cols;

    //namedWindow(window1, CV_WINDOW_AUTOSIZE);
    //namedWindow(window2, CV_WINDOW_AUTOSIZE);

    Mat query_mat;

    // Get input for subject
    FILE * input_file = fopen(argv[3],"r");

    int x,y;
    vector<Point> interested_points;
    {
        vector<vector<float> > query_points;

        while (fscanf(input_file,"%d%d",&x,&y) == 2){
            interested_points.push_back(Point(x,y));

            // get the click point in x,y coordinate
            vector<float> query_point;
            query_point.resize(descriptor_length);
            for (int i=0; i<descriptor_length; i++){
                query_point[i] = dense_descriptors_im2.at<float>(y*im2.cols+x, i);
            }

            query_points.push_back( query_point );
        }

        // now we need to build our query mat
        query_mat = Mat(query_points.size(), descriptor_length, CV_32F);
        for (int i=0; i<query_points.size(); i++){
            for (int j=0; j<descriptor_length; j++){
                query_mat.at<float>(i,j) = query_points[i][j];
            }
        }

        // and voila, query_points is free
    }

    //*****************************
    // perform search
    int num_points = query_mat.rows;

    Mat resulting_indices(num_points, num_knn_to_search,CV_32S);
    Mat resulting_distances(num_points, num_knn_to_search, CV_32F);

    cv::flann::SearchParams search_params;
    search_tree->knnSearch(query_mat, resulting_indices, resulting_distances, num_knn_to_search, search_params);

    // find min, max distance
    double min_distance, max_distance;
    minMaxLoc(resulting_distances, &min_distance, &max_distance);

    for (int i=0; i<num_points; i++){
        int found_index = resulting_indices.at<int>(i,0);
        float found_distance = resulting_distances.at<float>(i,0);

        Point found_point;
        found_point.x = found_index % im.cols;
        found_point.y = found_index / im.cols;

        //std::cout << "Found point at " << found_point.x << ", " << found_point.y << " " << found_distance << " away"<< std::endl;

        // draw original
        circle(im2, interested_points[i], 1, CV_RGB(0,255,0));

        // draw matched
        int scaled_radius = -(found_distance - min_distance)/(max_distance-min_distance) * (255-0)  + 255;
        circle(im, found_point, 1, CV_RGB((scaled_radius<1)?1:scaled_radius,0,0));
    }
    //imshow(window2,im2);
    //imshow(window1,im);

    //waitKey(0);

    std::cout << "saving images" << std::endl;
    imwrite("im1.png",im);
    imwrite("im2.png",im2);

    delete search_tree;

    fclose(input_file);
    return 0;
}
