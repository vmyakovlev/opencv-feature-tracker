#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

#include "misc.h"
#include "feature_detector.h"
#include "descriptor_match.h"
#include "window_pair.h"

int main (int argc, char ** argv){
    if (argc < 3){
        cout << "Usage: %prog [options] input_video.avi homography_points.txt\n";
        exit(-1);
    }
    //**************************************************************
    // PARAMETERS
    const string input_video_filename = string(argv[1]);
    const string homography_points_filename = string(argv[2]);

    //**************************************************************
    // Get the homography which brings coordinates in the image ground plane to world ground plane

    // Load points from file
    Mat homography_points = loadtxt(homography_points_filename);
    Mat image_points = homography_points.rowRange(0,4);
    Mat world_points = homography_points.rowRange(4,8);

    Mat homography_matrix = findHomography(image_points, world_points);
    std::cout << "Homography matrix: " << std::endl;
    print_matrix<float>(homography_matrix);

    //**************************************************************
    // PREPARE TOOLS FOR EXTRACTING FEATURES
    ShiTomashiFeatureDetector feature_detector;
    KLTTracker feature_matcher;

    vector<KeyPoint> key_points;

    //**************************************************************
    // GRAB SOME INFORMATION ABOUT THE VIDEO

    // Load the video
    VideoCapture video_capture(input_video_filename);
    if (!video_capture.isOpened()){
        cerr << "ERROR: Cannot open input video\n";
        exit(-3);
    }

    // Grab the first frame
    Mat a_frame;
    video_capture >> a_frame;
    if (a_frame.empty()){
        cerr << "ERROR: Cannot grab a frame from the video\n";
        exit(-2);
    }
    int frame_width = a_frame.cols,
        frame_height = a_frame.rows;

    //**************************************************************
    // SOME WINDOWS FOR VISUALIZATION
    const string window1("Frame");
    //namedWindow(window1);

    int keypressed_code;

    //**************************************************************
    // GO THROUGH THE ENTIRE VIDEO AND BUILD THE SPATIAL TEMPORAL GRAPH
    Mat prev_frame; // the previous frame in grayscale
    Mat next_frame;
    vector<Point2f> new_points;
    vector<int> old_points_indices;

    // Initialize our previous frame
    video_capture.grab();
    video_capture.retrieve(a_frame);
    cvtColor(a_frame,prev_frame, CV_RGB2GRAY);
    feature_detector.detect(prev_frame, key_points);
    feature_matcher.add(prev_frame, key_points);

    while (video_capture.grab()){
        video_capture.retrieve(a_frame);
        cvtColor(a_frame,next_frame, CV_RGB2GRAY);

        // Since we are using KLT, we will use KLTTracker directly without
        // switching to using DescriptorExtractor

        // Find these points in the next frame
        feature_matcher.search(next_frame, new_points, old_points_indices);

        // Show the frames (with optional annotations)
        WindowPair window_pair(prev_frame,next_frame,window1);
        // Draw these keypoints
        for (int i=0; i<old_points_indices.size(); ++i){
            window_pair.DrawArrow(key_points[old_points_indices[i]].pt, new_points[i], CV_RGB(255,0,0));
        }

        // Handle the events by waiting for a key
        keypressed_code = window_pair.Show();
        if (keypressed_code == 27) // ESC key
            break;

        // Go to the next frame
        next_frame.copyTo(prev_frame);
        feature_detector.detect(prev_frame, key_points);
        feature_matcher.add(prev_frame, key_points);
    }

    cout << "Done\n";

    return 0;
}
