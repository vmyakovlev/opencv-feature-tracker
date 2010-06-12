#include "blob_feature.h"

using namespace cv;

static unsigned int DESCRIPTOR_NUM_DIMENSION = 6;

/** \brief Extract the descriptor from our blob

    For blob feature, we will use the following information
  - color (mean color) (1 or 3; we will support only 3 for now) (perhaps we should perform PCA to find the top eigen colors)
  - location (2)
  - area (NOT included, used as weight for EMD instead)
  - fraction of: area / area of all blobs (1)

  */
void BlobDescriptorExtractor::compute(const cv::Mat& image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors)
{
    CV_Assert(image.channels() == 3)

    int num_keypoints = keypoints.size();

    float area_sum = 0;
    for (int i=0; i<num_keypoints; i++){
        area_sum += keypoints[i].size;
    }

    descriptors = Mat(num_keypoints, DESCRIPTOR_NUM_DIMENSION, CV_32F );

    // NOTE: Perhaps we can use MatIterator to have faster access to lines in the descriptor matrix
    for (int i=0; i<num_keypoints; i++){
        // compute blob mean color
        descriptors(Range(i,i+1), Range(0,3)) = blob_mean_color(image, keypoints[i]);

        // location
        descriptors.at<float>(i,3) = keypoints[i].pt.x;
        descriptors.at<float>(i,4) = keypoints[i].pt.y;

        // area fraction
        descriptors.at<float>(i,5) = keypoints[
                i].response / area_sum;
    }
}

/** \brief Compute mean color of at this keypoint

    The choice of having the color returned as a horizontal vector is so that you can build
    a descriptor Mat easily.

    \todo Use CIELa*b* color instead of RGB
    \return a (1,3) matrix with the mean color of the blob
*/
Mat BlobDescriptorExtractor::blob_mean_color(const Mat & image, const KeyPoint & keypoint){
    // first we need to get the upright bounding box for to start our scanline
    // NOTE: we lose one of the size dimension when we coverted our blob to keypoint
    Rect bounding_box = RotatedRect(keypoint.pt, Size(keypoint.size, keypoint.size), keypoint.angle).boundingRect();

    // keypoint rectangle in its coordinate system
    int half_size  = keypoint.size / 2;
    Rect keypoint_rect(keypoint.pt.x-half_size, keypoint.pt.y-half_size, keypoint.size, keypoint.size);

    Mat_<float> rotation_matrix = getRotationMatrix2D(Point2f(0,0), keypoint.angle, 1);
    Mat_<float> point(3,1);
    point[2][0] = 1; // this point needs to be in homogeneous coordinate
    Mat_<float> rotated_point(2,1);
    Point rotated_point2; // rotated point in Point_<int> format

    Mat accum(1,1, CV_64FC3);

    int num_inside = 0;
    // start line scanning
    for (int i=0; i<bounding_box.height; i++){
        for (int j=0; j<bounding_box.width; j++){
            // this point is inside when its rotated point is inside keypoint_rect
            point[0][0] = j;
            point[1][0] = i;

            rotated_point = rotation_matrix * point;

            // some lossy conversion
            rotated_point2.x = rotated_point[0][0];
            rotated_point2.y = rotated_point[1][0];

            if (!keypoint_rect.contains(rotated_point2))
                continue;

            // Ok, it is inside, accumulate this
            num_inside++;
            accumulate(Mat(image,Rect(i,j,1,1)),accum);
        }
    }

    // get mean value from accumulated
    Mat mean_color(1,3,CV_32F);
    float normalization_term = 1.0 / num_inside;

    Vec3d & accum_color = accum.at<Vec3d>(0,0);
    mean_color.at<float>(0,0) = accum_color[0] * normalization_term;
    mean_color.at<float>(0,1) = accum_color[1] * normalization_term;
    mean_color.at<float>(0,2) = accum_color[2] * normalization_term;

    return mean_color;
}

/** \brief Distance between two blob feature extracted

  The expected feature field descriptions are
  - color1
  - color2
  - color3
  - location1
  - location2
  - area_fraction

  \param feature1 Feature vector 1
  \param feature2 Feature vector 2
  \param user_data Extra user data passed by cvCalcEMD2

  \see cvCalcEMD2, compute
*/
float blob_distance(const float * feature1, const float * feature2, void * user_data){
    // normalized distance (everything goes into around [0,1]
    double diagonal_image_length = *(double*)user_data;

    float distance = 1.0/(255)* feature1[0] - 1.0/(255)* feature2[0] +
                        1.0/(255)* feature1[1] - 1.0/(255)* feature2[1] +
                        1.0/(255)* feature1[2] - 1.0/(255)* feature2[2] +
                        (feature1[3] - feature2[3]) +
                        (feature1[4] - feature2[4]) +
                        0.5 * feature1[5] - 0.5 * feature2[5];

    return distance;
}

