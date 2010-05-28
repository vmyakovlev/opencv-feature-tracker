#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include "opencv_io_extra.h"
#include "misc.h"
#include "daisy_feature.h"

using std::string;
using std::vector;
using namespace cv;

/** \file PCAConstruct.cxx Perform PCA on a set of features extracted for a specific subject

  Please note that this program will require quite a bit of memory to run

  First you need a file PCAConstructParameters.txt that has
  1 0 60
  2 0 60
  3 2 62

  Each line is: <subject_id> <start_frame> <end_frame>

  Then run

  awk '{print "cd Subject"$1" ; ~/build/ACV-HW2/PCAConstruct "$1" "$2" "$3" ~/Desktop/subject"$1"_pca.yml";cd ..}' PCAConstructParameters.txt > PCAConstruct.txt

  Now, you can do

  bash PCAConstruct.txt

  If you really have lots of memory and want parallel processing, you can use ppss like this

  ppss -f PCAConstruct.txt -c 'bash -c $ITEM'


*/
int main(int argc, char ** argv){
    if (argc<6){
        printf("Usage: %s [options] subject_id start_frame_no end_frame_no num_pc_to_keep output_file.cvmat\n",argv[0]);
        exit(-1);
    }

    const int subject_id = atoi(argv[1]);
    const int start_frame_no = atoi(argv[2]);
    const int end_frame_no = atoi(argv[3]);
    const int num_principle_components = atoi(argv[4]);

    // First loop to figure out how big do we need the final matrix to be
    std::cout << "Tallying the total number of features we will have " << std::endl;

    char query_points_filename[256];
    int total_num_features = 0;
    vector<int> feature_count;
    for (int frame_no=start_frame_no; frame_no<end_frame_no; frame_no++){
        // Get filenames according to our convention
        sprintf(query_points_filename, "query_points_%d_%d.txt", frame_no, subject_id);

        Mat query_points;
        try{
            query_points = loadtxt(query_points_filename);
        } catch (cv::Exception& e) {
            printf("Warning: %s doesn't exist\n", query_points_filename);

            feature_count.push_back(0);
            continue; // can't find this file, let's skip it
        }

        //std::cout << query_points.rows << " ";

        // tally feature counts
        feature_count.push_back(query_points.rows);
        total_num_features = total_num_features + query_points.rows;
    }
    std::cout << std::endl;

    std::cout << "Allocating memory for " << total_num_features << " features" << std::endl;
    Mat all_descriptors(total_num_features, 200, CV_32F);

    DaisyDescriptorExtractor descriptor_extractor;

    // Second loop, extract DAISY features and save directly into the allocated space
    std::cout << "Extracting DAISY signatures" << std::endl;

    // extract DAISY signatures for each frame
    int row_offset = 0;
    char image_filename[256];
    Mat dense_descriptors_im;
    for (int frame_no=start_frame_no; frame_no<end_frame_no; frame_no++){
        // Get filenames according to our convention
        sprintf(image_filename, "frame_%04d.jpg", frame_no);
        sprintf(query_points_filename, "query_points_%d_%d.txt", frame_no, subject_id);

        // Debug
        std::cout << "Processing " << image_filename << " and " << query_points_filename << std::endl;

        Mat query_points;
        // Load query points
        try{
            query_points = loadtxt(query_points_filename);
        } catch (cv::Exception& e) {
            printf("Warning: %s doesn't exist\n", query_points_filename);
            continue; // can't find this file, let's skip it
        }

        // read in the image
        Mat gray_im = imread(image_filename,0);

        // compute the dense descriptors from this image
        descriptor_extractor.compute_dense(gray_im, dense_descriptors_im);

        // prepare to get only the features that we want (from query_points)
        Mat row_indexes_float = query_points.col(0) + query_points.col(1) * gray_im.cols;
        Mat row_indexes;
        row_indexes_float.convertTo(row_indexes, CV_32S);

        //cvSave("/Users/dchu/Desktop/row_indexes.cvmat", &CvMat(row_indexes));

        // get only the features we are interested in
        for (int j=0; j<row_indexes.rows; j++){
            int source_row_index = row_indexes.at<int>(j,0);

            // NOTE: we need to offset the row because of how we are storing descriptors
            //       We are simply vertical-stack all descriptors together.
            Mat target_row = all_descriptors.row(row_offset + j);

            dense_descriptors_im.row(source_row_index).copyTo(target_row);
        }
        // update row offset
        row_offset = row_offset + feature_count[frame_no - start_frame_no];
    }
    std::cout << std::endl;

    // Time to compute PCA
    std::cout << "Compute PCA" << std::endl;
    PCA pca(all_descriptors, Mat(), CV_PCA_DATA_AS_ROW, num_principle_components);

    // Save this PCA for later recognition
    std::cout << "Saving PCA results" << std::endl;
    string storage_filename(argv[5]);
    FileStorage fs(storage_filename, FileStorage::WRITE);
    write(fs,pca);

    // Done
    std::cout << "Done computing PCA" << std::endl;
    return 0;
}
