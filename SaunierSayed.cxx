#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

#include "misc.h"
#include "feature_detector.h"

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
    namedWindow(window1);

    int keypressed_code;

    //**************************************************************
    // GO THROUGH THE ENTIRE VIDEO AND BUILD THE SPATIAL TEMPORAL GRAPH
    Mat gray_frame; // the frame in grayscale
    while (video_capture.grab()){
        video_capture.retrieve(a_frame);
        cvtColor(a_frame,gray_frame, CV_RGB2GRAY);

        // Initialize FeatureDetector
        feature_detector.detect(gray_frame, key_points);

        // Draw these keypoints
        for (int i=0; i<key_points.size(); ++i){
            circle(a_frame, key_points[i].pt, 1, CV_RGB(255,0,0));
        }


        // Show the frame (with optional annotations)
        imshow(window1, a_frame);

        // Handle the events by waiting for a key
        keypressed_code = waitKey(0);
        if (keypressed_code == 27) // ESC key
            break;
    }

    cout << "Done\n";

    return 0;
}
