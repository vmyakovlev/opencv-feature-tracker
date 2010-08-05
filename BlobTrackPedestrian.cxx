#include <iostream>
#include <fstream>
#include <string>
#include <gflags/gflags.h>

// Usage of new OpenCV include files
// Only work with OpenCV SVN
#include <opencv2/opencv.hpp>
#include <opencv2/video/blobtrack2.hpp>

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

    // Initialize our previous frame
    video_capture.grab();
    video_capture.retrieve(a_frame);
    cvtColor(a_frame,prev_frame, CV_RGB2GRAY);

    std::vector<Blob> detected_blobs;
    int current_num_frame = 0;
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

         // Debug: Visualize detected blobs
         blob_image.setTo(Scalar(0));
         char custom_message[50];
         for (size_t i=0; i<detected_blobs.size(); i++){
             sprintf(custom_message, "%d",i );
             detected_blobs[i].DrawTo(blob_image, custom_message);
         }
         imshow(blob_window, blob_image);

         // Handle GUI events by waiting for a key
         keypressed_code = cv::waitKey();
         if (keypressed_code == 27){ // ESC key
             break;
         }

        // some indication of stuff working
        cout << "\r" << current_num_frame << flush;
        current_num_frame++;
    }
    printf("\n");

    cout << "Done: BlobTrackPedestrian\n";

    return 0;
}
