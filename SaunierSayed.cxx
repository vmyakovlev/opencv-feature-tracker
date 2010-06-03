#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <string>
#include <gflags/gflags.h>
using namespace std;
using namespace cv;

#include "misc.h"
#include "feature_detector.h"
#include "descriptor_match.h"
#include "window_pair.h"
#include "SaunierSayed_feature_grouping.h"

DEFINE_bool(homography_point_correspondence, false, "The homography file contains correspondences instead of the homography matrix");
DEFINE_bool(debug_gui, true, "Use GUI to debug");

void convert_to_world_coordinate(const vector<Point2f> & points_in_image_coordinate, const Mat & homography_matrix, vector<Point2f> * points_in_world_coordinate){
    points_in_world_coordinate->clear();
    points_in_world_coordinate->resize(points_in_image_coordinate.size());
    Point2f temp_point;

    Mat points_in_image_coordinate_mat(points_in_image_coordinate, false); // sharing data
    Mat points_in_world_coordinate_mat(*points_in_world_coordinate, false); // sharing data so output is written to the right place
    perspectiveTransform(points_in_image_coordinate_mat, points_in_world_coordinate_mat, homography_matrix);
}

int main (int argc, char ** argv){
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4){
        cout << "Usage: %prog [options] input_video.avi homography.txt output.txt\n";
        exit(-1);
    }

    //**************************************************************
    // PARAMETERS
    const string input_video_filename = string(argv[1]);
    const string homography_points_filename = string(argv[2]);
    const string output_filename = string(argv[3]);

    //**************************************************************
    // Get the homography which brings coordinates in the image ground plane to world ground plane

    // We allow two different ways to pass in homography information
    // 1 - Read from a file (default)
    // 2 - Calculate from points correspondence
    Mat homography_matrix;

    if (FLAGS_homography_point_correspondence){
        // Load points from file
        Mat homography_points = loadtxt(homography_points_filename);
        Mat image_points = homography_points.rowRange(0,4);
        Mat world_points = homography_points.rowRange(4,8);

        homography_matrix = findHomography(image_points, world_points);
    } else {
        // simply read the homography matrix from file
        homography_matrix = loadtxt(homography_points_filename);
    }

    // Verbose debug printing
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
    const string window3("Original");
    const string window2("World Coordinate");
    namedWindow(window2);
    namedWindow(window3);

    int keypressed_code;

    //**************************************************************
    // GO THROUGH THE ENTIRE VIDEO AND BUILD THE SPATIAL TEMPORAL GRAPH
    Mat prev_frame; // the previous frame in grayscale
    Mat next_frame;
    vector<Point2f> new_points;
    vector<int> old_points_indices;
    SaunierSayed::TrackManager feature_grouper(2,4,25,true);

    // Initialize our previous frame
    video_capture.grab();
    video_capture.retrieve(a_frame);
    cvtColor(a_frame,prev_frame, CV_RGB2GRAY);
    feature_detector.detect(prev_frame, key_points);
    feature_matcher.add(prev_frame, key_points);

    // Add detected points from the first frame to our feature grouper
    vector<Point2f> frame_points;
    vector<Point2f> frame_points_in_world;
    vector_one_to_another(key_points, frame_points);
    convert_to_world_coordinate(frame_points, homography_matrix, &frame_points_in_world);
    feature_grouper.AddPoints(frame_points_in_world);

    // unwarp the image using this homography matrix and shows it
//    Mat warpedImage;
//    warpPerspective(a_frame, warpedImage, homography_matrix, Size(512,512),
//                    //CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
//                    CV_WARP_INVERSE_MAP | CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
//    imshow(window2, warpedImage);
//    imshow(window3, a_frame);
//    waitKey(0);

    while (video_capture.grab()){
        video_capture.retrieve(a_frame);
        cvtColor(a_frame,next_frame, CV_RGB2GRAY);
        imshow(window3, a_frame);

        // Since we are using KLT, we will use KLTTracker directly without
        // switching to using DescriptorExtractor

        // Find these points in the next frame
        feature_matcher.search(next_frame, new_points, old_points_indices);

        // Tally these new points into our graphical model
        feature_grouper.UpdatePoints(new_points, old_points_indices);

        // *********************************************************
        // Show GUI for debugging purposes
        if (FLAGS_debug_gui){
            // Show the frames (with optional annotations)
            WindowPair window_pair(prev_frame,next_frame,window1);
            // Draw these keypoints
            for (int i=0; i<old_points_indices.size(); ++i){
                window_pair.DrawArrow(key_points[old_points_indices[i]].pt, new_points[i], CV_RGB(255,0,0));
            }

            // Handle GUI events by waiting for a key
            keypressed_code = window_pair.Show();
            if (keypressed_code == 27){ // ESC key
                // take a screenshot
                imwrite("screenshot.png", a_frame);
                break;
            }
        }

        // Go to the next frame
        next_frame.copyTo(prev_frame);
        feature_detector.detect(prev_frame, key_points);
        feature_matcher.add(prev_frame, key_points);

        vector_one_to_another(key_points, frame_points);
        convert_to_world_coordinate(frame_points, homography_matrix, &frame_points_in_world);
        feature_grouper.AddPossiblyDuplicatePoints(frame_points_in_world);

        // some indication of stuff working
        std::cout << ".";
    }

    //TODO: Collect track information and Report
    SaunierSayed::ConnectedComponents components = feature_grouper.GetConnectedComponents();

    // Write components to disk
    ofstream output_file;
    output_file.open(output_filename.c_str());

    SaunierSayed::ConnectedComponent component;
    for (int i=0; i<components.size(); ++i){
        component = components[i];

        output_file << i << " ";
        for (int j=0; j<component.size(); ++j){
            output_file << component[j].id << " ";
        }

        output_file << std::endl;
    }

    output_file.close();

    cout << "Done\n";

    return 0;
}
