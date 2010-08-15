/** \file BlobTrackPedestri
  A demo that performs blob tracking on pedestrians. Documentations including different outputs in different versions
  of this code can be found at https://code.google.com/p/opencv-feature-tracker/wiki/BlobTrackDemo
*/
#include <iostream>
#include <fstream>
#include <string>
#include <gflags/gflags.h>

// Usage of new OpenCV include files
// Only work with OpenCV SVN
#include <opencv2/opencv.hpp>
#include <opencv2/video/blobtrack2.hpp>

#include "misc.h"

using cv::Mat;
using namespace std;

DEFINE_bool(debug_gui, true, "Use GUI to debug");

int main (int argc, char ** argv){
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 3){
        cout << "Usage: %prog [options] input_video.avi output_video.avi\n";
        exit(-1);
    }

    //**************************************************************
    // PARAMETERS
    const string input_video_filename = string(argv[1]);
    const string output_filename = string(argv[2]);

    //**************************************************************
    // PRINT OUT PARAMETERS
    cout << "Debug GUI: " << FLAGS_debug_gui << endl;

    //**************************************************************
    // PREPARE TOOLS FOR BACKGROUND SUBTRACTION
    cv::BackgroundSubtractorMOG background_subtractor;
    cv::BlobDetector blob_detector;
    cv::BlobTrajectoryTracker blob_tracker;
    cv::BlobMatcherWithTrajectory blob_matcher(&blob_tracker);

    //**************************************************************
    // GRAB SOME INFORMATION ABOUT THE VIDEO

    // Load the video
    cv::VideoCapture video_capture(input_video_filename);
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
    const string blob_window("Blobs");
    const string original_video("Original");
    const string fg_window("Foreground");
    cv::namedWindow(blob_window);
    cv::namedWindow(fg_window);
    cv::namedWindow(original_video);

    int keypressed_code;

    //**************************************************************
    // GO THROUGH THE ENTIRE VIDEO AND BUILD THE SPATIAL TEMPORAL GRAPH
    Mat prev_frame; // the previous frame in grayscale
    Mat next_frame;
    Mat fg_mask; //foreground mask detected in each frame
    Mat blob_image = cv::Mat::zeros(a_frame.size(), CV_8UC3); // for visualizing the detected blobs
    std::vector<Blob> detected_blobs;

    // Initialize with the first 2 frames
    // Instead of using an if within the while loop, we will re-type some code here
    video_capture.grab();
    video_capture.retrieve(a_frame);
    cvtColor(a_frame,prev_frame, CV_RGB2GRAY);
    background_subtractor(a_frame, fg_mask);

    // We need two frames because with only one frame, we cannot subtract the background
    video_capture.grab();
    video_capture.retrieve(a_frame);
    background_subtractor(a_frame, fg_mask);
    detected_blobs = blob_detector(fg_mask, 1);
    blob_tracker.addTracks(detected_blobs);
    blob_tracker.nextTimeInstance();

    int current_num_frame = 0;
    std::vector<BlobTracker::id_type> matches; // The id of the found match using the matcher
    std::map<BlobTracker::id_type, Blob> detected_blobs_with_matched_ids;
    while (video_capture.grab()){
        video_capture.retrieve(a_frame);
        cvtColor(a_frame,next_frame, CV_RGB2GRAY);
        imshow(original_video, a_frame);

        // Use the BackgroundSubtractor to subtract the background
        background_subtractor(a_frame, fg_mask);

        // Debug: Show foreground window
        imshow(fg_window, fg_mask);

        // Use Blob Detector to detect the blobs
        detected_blobs = blob_detector(fg_mask, 1);

        // Use Blob Matcher to match these new blobs to existing blobs
        blob_matcher.match(next_frame, detected_blobs, matches);

        // Combine matches and detected_blobs into a map
        detected_blobs_with_matched_ids = vec_vec_to_map(matches, detected_blobs);

        // Debug: Visualize detected blobs
        blob_image.setTo(Scalar(0));
        char custom_message[50];
        std::map<BlobTracker::id_type, Blob>::const_iterator matched_blobs_iterator = detected_blobs_with_matched_ids.begin();
        for (; matched_blobs_iterator!=detected_blobs_with_matched_ids.end();
               matched_blobs_iterator++){
            sprintf(custom_message, "%d", (*matched_blobs_iterator).first );
            (*matched_blobs_iterator).second.DrawTo(blob_image, custom_message);
        }
        imshow(blob_window, blob_image);

        // Update the tracker with new information
        blob_tracker.updateTracks(detected_blobs_with_matched_ids);

        // Handle GUI events by waiting for a key
        keypressed_code = cv::waitKey();
        if (keypressed_code == 27){ // ESC key
            break;
        }

        // some indication of stuff working
        cout << "\r" << current_num_frame << flush;

        // advance to the next frame
        current_num_frame++;
        blob_tracker.nextTimeInstance();

    }
    printf("\n");

    cout << "Done: BlobTrackPedestrian\n";

    return 0;
}
