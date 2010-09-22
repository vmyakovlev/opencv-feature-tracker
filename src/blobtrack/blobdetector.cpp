/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

Implement a blob detector

*/

/*///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//*/
#include <opencv2/video/blobtrack2.hpp>
#include <opencv2/opencv.hpp>
#include "misc.h"
namespace cv{
    BlobDetector::BlobDetector(){}

    /** \brief Perform blob detection from given foreground mask image

      \param[in] input_foreground_mask_image single channel image where > 128 is foreground and <= 128 is background
      \param close_holes How many times to perform hole closing operations. Becareful of the size of your expected
                        foreground blobs. Typically 1 will remove spurious noises.
    */
    std::vector<Blob> BlobDetector::operator()(const cv::Mat & input_foreground_mask_image, int close_holes/* = 1*/) const {
        // input check
        CV_Assert(input_foreground_mask_image.channels() == 1)

        std::vector<Blob> found_blobs;

        // threshold the foreground mask image
        Mat foreground_mask;
        threshold(input_foreground_mask_image, foreground_mask, 128, 255, THRESH_BINARY);

        // Perform morphological operations to remove those small things
        if (close_holes > 0){
            Mat temp_foreground_mask;

            // close holes
            cv::erode(foreground_mask, temp_foreground_mask, Mat(), Point(-1,-1), close_holes);
            cv::dilate(temp_foreground_mask, foreground_mask, Mat(), Point(-1,-1), close_holes);

            // close gaps
            cv::dilate(temp_foreground_mask, foreground_mask, Mat(), Point(-1,-1), close_holes);
            cv::erode(foreground_mask, temp_foreground_mask, Mat(), Point(-1,-1), close_holes);
        }

        // Debug: show this image
        const string window = "Debug: BlobDetector";
        cv::namedWindow(window);
        cv::imshow(window, foreground_mask);

        // find the contours
        // since findContours do not support Point2f, we need a little work
        std::vector<std::vector<Point> > contour_points_2i;
        findContours(foreground_mask, contour_points_2i, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        // assign these contours into blobs
        std::vector<Point2f> points;
        for (size_t i=0; i<contour_points_2i.size(); i++){
            vector_one_to_another(contour_points_2i[i], points);
            found_blobs.push_back( Blob(points) );
        }

        return found_blobs;
    }
}
