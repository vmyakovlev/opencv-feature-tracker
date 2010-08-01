#include <opencv2/video/blobdetector.hpp>
#include <opencv2/opencv.hpp>

namespace cv{
    BlobDetector::BlobDetector(){}

    /** \brief Perform blob detection from given foreground mask image

      \param[in] input_foreground_mask_image single channel image where > 128 is foreground and <= 128 is background
    */
    std::vector<Blob> BlobDetector::operator()(const cv::Mat & input_foreground_mask_image) const {
        // input check
        CV_Assert(input_foreground_mask_image.channels() == 1)

        std::vector<Blob> found_blobs;

        // threshold the foreground mask image
        Mat foreground_mask;
        threshold(input_foreground_mask_image, foreground_mask, 128, 255, THRESH_BINARY);

        // find the contours
        std::vector<std::vector<Point> > contour_points;
        findContours(foreground_mask, contour_points, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        // assign these contours into blobs
        for (size_t i=0; i<contour_points.size(); i++){
            found_blobs.push_back( Blob(contour_points[i]) );
        }
    }
}
