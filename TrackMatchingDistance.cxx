#include <cv.h>
#include <highgui.h>
#include <string>
#include <iostream>
#include <stdexcept>

using namespace cv;
using namespace std;

#include "daisy_feature.h"
#include "misc.h"

/** \brief Accumulate the descriptors calculated
*/
void AccumulateTrackFeatures(const Mat & new_descriptor, Mat * im1_sum_descriptor, int * im1_descriptor_count){
    for(int i=0; i<new_descriptor.rows; ++i){
        (*im1_sum_descriptor).row(0) += new_descriptor.row(i);
    }

    // Increase the count of descriptors
    (*im1_descriptor_count) += im1_sum_descriptor->rows;
}

void PrintAccumulatedTrackFeatures(const Mat & track_features){
    const float * feature = track_features.ptr<float>(0);
    for (int i=0; i<track_features.cols; ++i){
        cout << *feature << " ";
        feature++;
    }
    cout << endl;
}

/** \file TrackMatchingDistance.cxx
  For each track, read the track result file and compute features for these tracks. The features are linearly combined (L1). One
  then has two float values per track. The difference between these two float values are taken as the cost
  that connects these two tracks.
*/
int main(int argc, char** argv){
    string usage_string = "[options] track1 track2 tracking_result.txt";

    if (argc<4){
        cout << usage_string << endl;
        exit(1);
    }

    int track1_no = atoi(argv[1]);
    int track2_no = atoi(argv[2]);

    // Open tracking result file for reading
    FILE * tracking_results_file = fopen(argv[3], "r");

    if (!tracking_results_file){
        cerr << "Cannot open tracking result file: " << argv[3] << endl;
    }

    ///////// PARAMETERS
    int track_offset = 5000;

    DaisyDescriptorExtractor daisy;
    int feature_length = daisy.feature_length();
    Mat im1_sum_descriptor = Mat::zeros(1, feature_length, CV_32F);
    Mat im2_sum_descriptor = Mat::zeros(1, feature_length, CV_32F);
    int im1_descriptor_count = 0;
    int im2_descriptor_count = 0;

    // Read through the result file and find only the entries for the tracks we are interested in
    int frame_no, track_no;
    float x,y,w,h;
    Mat im;
    char image_filename[256], query_filename[256];
    Mat descriptors; // temporary (used for computing descriptors)
    Mat query_points;
    while (fscanf(tracking_results_file, "%i%i%f%f%f%f",&frame_no, &track_no,&x,&y,&w,&h) == 6){
        if (track_no != track1_no && track_no != track2_no){
            cout << "+";
            fflush(stdout);
            continue;
        }
        cout << ".";
        fflush(stdout);

        // Load this image
        sprintf(image_filename, "frame_%04d.jpg", frame_no);
        sprintf(query_filename, "query_points_%d_%d.txt", frame_no, track_no);

        cout << "Reading image: " << image_filename << endl;
        im = imread(image_filename, 0); // gray-scale image only

        // Find interest points (corners)
        cout << "Find Interest Points: " << query_filename << endl;
        query_points = FindInterestPoints(im, query_filename);

        // Update size to the new number of query points
        descriptors = Mat(query_points.rows, descriptors.cols, descriptors.type());

        // Compute daisy feature
        cout << "Compute DAISY" << endl;
        daisy.compute(im, query_points, descriptors);

        // Accumulate all the descriptors for this track
        cout << "Accumulate" << endl;
        if (track_no == track1_no){
            AccumulateTrackFeatures(descriptors, &im1_sum_descriptor, &im1_descriptor_count);
        } else {
            AccumulateTrackFeatures(descriptors, &im2_sum_descriptor, &im2_descriptor_count);
        }
    }

    // Print out the accumulated features
    cout << "Feature count " << im1_descriptor_count << " " << im2_descriptor_count << endl;
    cout << "Feature 1:" << endl;
    Mat normalized_im1_sum_descriptor = im1_sum_descriptor/im1_descriptor_count;
    PrintAccumulatedTrackFeatures(normalized_im1_sum_descriptor);
    cout << "Feature 2:" << endl;
    Mat normalized_im2_sum_descriptor = im1_sum_descriptor/im2_descriptor_count;
    PrintAccumulatedTrackFeatures(normalized_im2_sum_descriptor);
    cout << endl;

    Mat difference = normalized_im1_sum_descriptor - normalized_im2_sum_descriptor;
    double norm_difference = norm(difference, CV_L1);
    cout << "Difference: " <<  norm_difference << endl;

    fclose(tracking_results_file);

    // Write results to file
    char output_filename[256];
    sprintf(output_filename,"%d_%d.txt", track1_no, track2_no);
    FILE * output_file = fopen(output_filename, "w");
    fprintf(output_file, "%g", norm_difference);
    fclose(output_file);

    return 0;
}
