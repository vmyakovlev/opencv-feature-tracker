#include "window_pair.h"
#include <cv.h>
#include <highgui.h>

using namespace cv;

WindowPair::WindowPair(const cv::Mat & im1, const cv::Mat & im2, const std::string & name)
{
    im_ = Mat::zeros(max(im1.rows,im2.rows), im1.cols + im2.cols, CV_8UC3);
    im2_offset = cv::Point(im1.cols,0);
    window_name_ = name;

    // copy image data to the right place
    cv::Mat im_im1 = cv::Mat(im_, cv::Range(0,im1.rows), cv::Range(0,im1.cols));
    cv::Mat im_im2 = cv::Mat(im_, cv::Range(0,im2.rows), cv::Range(im1.cols,im1.cols+im2.cols));

    // convert grayscale image as necessary
    cv::Mat im1_color, im2_color;
    if (im1.channels() == 1){
        cv::cvtColor(im1,im1_color, CV_GRAY2RGB);
    } else if (im1.channels() == 3) {
        im1_color = im1;
    } else {
        CV_Error(CV_StsBadSize, "input image 1 needs to be either single-channel or 3-channel");
    }

    if (im2.channels() == 1){
        cvtColor(im2,im2_color, CV_GRAY2RGB);
    } else if (im2.channels() == 3) {
        im2_color = im2;
    } else {
        CV_Error(CV_StsBadSize, "input image 2 needs to be either single-channel or 3-channel");
    }

    // Copy data to our internal image
    im1_color.copyTo(im_im1);
    im2_color.copyTo(im_im2);
}

/** \brief Draw an arrow from a point in im1 to a point in im2
  \param[in] im1_from the start point of the arrow in im1 coordinate
  \param[in] im2_to the end point of the arrow in im2 coordinate
*/
void WindowPair::DrawArrow(cv::Point im1_from, cv::Point im2_to, const cv::Scalar & color, int thickness, int lineType, int shift){
    cv::line(im_, im1_from, im2_offset + im2_to, color, thickness, lineType, shift);
}

void WindowPair::Show(int delay){
    cv::namedWindow(window_name_,CV_WINDOW_AUTOSIZE);
    cv::imshow(window_name_,im_);
    cv::waitKey(delay);
}

/** \brief Save output window pair image to filename
*/
void WindowPair::Save(const std::string & filename){
    cv::imwrite(filename, im_);
}

/** \brief Get access to the internal image
*/
cv::Mat WindowPair::get_image(){
    return im_;
}
