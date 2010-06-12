#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include "misc.h"
#include "opencv_io_extra.h"
#include "daisy_feature.h"

using std::string;
using std::vector;
using namespace cv;

/** \file PCARecognizeInterestPoints.cxx Perform PCA on a set of features extracted for a specific subject.
  The given query points are used to extract the target bounding box location
  Then 10 corners are detected from this bounding box and only 10 corners are used for matching to our PCAs

  \see PCAConstruct.cxx
*/
int main(int argc, char ** argv){
    if (argc<5){
        printf("Usage: %s [options] input_image.png query_points.txt pca_folder num_subjects_in_gallery [output.txt]\n",argv[0]);
        exit(-1);
    }

    const int num_gallery_subjects = atoi(argv[4]);
    const char * pca_folder = argv[3];

    //********************************************************************
    // Time to compute PCA
    std::cout << "Loading gallery PCAs" << std::endl;
    vector<PCA> gallery_PCAs;

    LoadGalleryPCAs(&gallery_PCAs, pca_folder, num_gallery_subjects);

    //********************************************************************
    // Compute DAISY features for query points
    std::cout << "Compute DAISY features for query points" << std::endl;

    // Load image as grayscale
    Mat gray_im = imread(string(argv[1]), 0);

    // Find interest points (corners)
    Mat query_points = FindInterestPoints(gray_im, argv[2]);

    DaisyDescriptorExtractor descriptor_extractor;
    Mat descriptors;
    descriptor_extractor.compute(gray_im, query_points, descriptors);

    //********************************************************************
    // Project into each PCA our test set
    std::cout << "Projecting into gallery PCA our test set" << std::endl;

    vector<vector<double> > reconstruction_error;
    reconstruction_error.resize(num_gallery_subjects);
    vector<double> cumulative_reconstruction_error;
    cumulative_reconstruction_error.resize(num_gallery_subjects);
    for (int j=0; j<descriptors.rows; j++){
        Mat vec = descriptors.row(j);

        // Debug: Printing this feature vector
//        std::cout << "> ";
//        for (int i=0; i<vec.cols; ++i){
//            std::cout << vec.at<float>(0,i) << " ";
//        }
//        std::cout << std::endl;

        for (int i=0; i<num_gallery_subjects; i++){
            Mat coeff;
            Mat reconstructed;

            // project the vector into PCA
            gallery_PCAs[i].project(vec, coeff);
            // reconstruct
            gallery_PCAs[i].backProject(coeff, reconstructed);

            // save reconstruction error
            reconstruction_error[i].push_back(norm(vec, reconstructed, NORM_L2));

            // accumulate the reconstruction error
            cumulative_reconstruction_error[i] += norm(vec, reconstructed, NORM_L2);
        }
    }

    //********************************************************************
    // Report
    std::cout << "Reporting:" << std::endl;
    std::cout << "Subject score0 score1 score2 score3 score4 score5 score6 score7 score8 score9 " << std::endl;
    for (int i=0; i<cumulative_reconstruction_error.size(); i++){
        // Sort scores per subject
        std::sort(reconstruction_error[i].begin(), reconstruction_error[i].end());

        // Verbose output
//        std::cout << "Subject " << i << " 's error = " << cumulative_reconstruction_error[i] // Total score
//                  << " " << reconstruction_error[i][4] // Median score
//                  << std::endl;

        // Easy to parse output
        std::cout << i+1 << " ";

        // also print the individual scores
        //std::cout << "< ";
        for (int j=0; j<reconstruction_error[i].size(); ++j){
            std::cout << reconstruction_error[i][j] << " ";
        }
        //std::cout << ">" << std::endl;

        std::cout << std::endl;
    }

    //********************************************************************
    // Writing results to output file
    if (argc > 5){
        FILE * f = fopen(argv[5], "w");

        if (f == NULL){
            printf("Cannot open file %s for writing\n", argv[5]);
            exit(-1);
        }

        for (int i=0; i<cumulative_reconstruction_error.size(); i++){
            fprintf(f, "%d ", i); // Print subject id
            for (int j=0; j<reconstruction_error[i].size(); ++j){
                fprintf(f, "%f ", reconstruction_error[i][j]);
            }
            fprintf(f, "\n"); // next line
        }

        fclose(f);

        printf("Done saving to file %s\n", argv[5]);
    }

    //********************************************************************
    // Done
    std::cout << "Done computing PCA" << std::endl;
    return 0;
}
