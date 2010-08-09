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