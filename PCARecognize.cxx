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

/** \file PCARecognize.cxx Perform PCA on a set of features extracted for a specific subject

  \see PCAConstruct.cxx
*/
int main(int argc, char ** argv){
    if (argc<5){
        printf("Usage: %s [options] input_image.png query_points.txt pca_folder num_subjects_in_gallery\n",argv[0]);
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

    Mat query_points = loadtxt(string(argv[2]));
    DaisyDescriptorExtractor descriptor_extractor;

    Mat gray_im = imread(string(argv[1]), 0);
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
    for (int i=0; i<cumulative_reconstruction_error.size(); i++){
        std::cout << "Subject " << i << " 's error = " << cumulative_reconstruction_error[i] << std::endl;
    }

    //********************************************************************
    // Done
    std::cout << "Done computing PCA" << std::endl;
    return 0;
}
