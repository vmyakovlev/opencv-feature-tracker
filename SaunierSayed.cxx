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
#include "feature_grouper_visualizer.h"

DEFINE_bool(homography_point_correspondence, false, "The homography file contains correspondences instead of the homography matrix");
DEFINE_bool(debug_gui, true, "Use GUI to debug");
DEFINE_bool(log_tracks_info, false, "Log most information about tracks as time progress (LOTS OF disk space required)");
DEFINE_uint64(min_frames_tracked, 4, "Minimum number of frames tracked before it is activated");
DEFINE_double(min_distance_moved_required, 3, "Minimum number of frames tracked before it is activated");
DEFINE_double(maximum_distance_activated, 20, "When activated, how far around the point do we search for points to add?");
DEFINE_double(segmentation_threshold, 200, "How much do we allow max_distance - min_distance to vary before an edge is severe.");

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
    // PRINT OUT PARAMETERS
    cout << "Compute homograph from points: " << FLAGS_homography_point_correspondence << endl;
    cout << "Debug GUI: " << FLAGS_debug_gui << endl;
    cout << "Logging tracks info: " << FLAGS_log_tracks_info << endl;
    cout << "Min frames tracked: " << FLAGS_min_frames_tracked << endl;
    cout << "Min distance moved required: " << FLAGS_min_distance_moved_required << endl;
    cout << "Max distance activated: " << FLAGS_maximum_distance_activated << endl;
    cout << "Segmentation threshold: " << FLAGS_segmentation_threshold << endl;

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
    vector<int> assigned_ids;
    vector<int> matched_track_ids;

    SaunierSayed::TrackManager feature_grouper(
            FLAGS_min_frames_tracked,
            FLAGS_min_distance_moved_required,
            FLAGS_maximum_distance_activated,
            FLAGS_segmentation_threshold,
            FLAGS_log_tracks_info);

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
    feature_grouper.AddPoints(frame_points_in_world, &assigned_ids);

    // unwarp the image using this homography matrix and shows it
//    Mat warpedImage;
//    warpPerspective(a_frame, warpedImage, homography_matrix, Size(512,512),
//                    //CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
//                    CV_WARP_INVERSE_MAP | CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
//    imshow(window2, warpedImage);
//    imshow(window3, a_frame);
//    waitKey(0);

    SaunierSayed::FeatureGrouperVisualizer visualizer(homography_matrix, &feature_grouper);

    while (video_capture.grab()){
        video_capture.retrieve(a_frame);
        cvtColor(a_frame,next_frame, CV_RGB2GRAY);
        imshow(window3, a_frame);
        visualizer.NewFrame(a_frame);

        // Since we are using KLT, we will use KLTTracker directly without
        // switching to using DescriptorExtractor

        // Find these points in the next frame
        feature_matcher.search(next_frame, new_points, old_points_indices);
        convert_to_world_coordinate(new_points, homography_matrix, &frame_points_in_world);

        // Map these old indices to the ids of the tracks
        matched_track_ids = indexing(assigned_ids, old_points_indices);

        // Tally these new points into our graphical model
        feature_grouper.UpdatePoints(frame_points_in_world, matched_track_ids);
        visualizer.Draw();

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
        assigned_ids.clear();
        feature_grouper.AddPossiblyDuplicatePoints(frame_points_in_world, &assigned_ids);

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
