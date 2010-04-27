#ifndef WINDOW_PAIR_H_
#define WINDOW_PAIR_H_

#include <cv.h>
#include <string>

/** \class WindowPair
    A window that manage two image put side by side. Has methods that allow annotations
    of points on each image. Good for showing correspondence between two image.
*/
class WindowPair{
public:
    WindowPair(const cv::Mat & im1, const cv::Mat & im2, const std::string & name);
    void DrawArrow(cv::Point im1_from, cv::Point  im2_to, const cv::Scalar & color, int thickness=1, int lineType=8, int shift=0);
    void Show(int delay=0);
    void Save(const std::string & filename);
    cv::Mat get_image();
private:
    cv::Mat im_;
    cv::Point im2_offset; //!< the offset that takes a point in im2 coordinate to window coordinate
    std::string window_name_;
};

#endif
