#include "daisy_feature.h"
#include <highgui.h>
DaisyDescriptorExtractor::DaisyDescriptorExtractor(double rad, int radq, int histq, int thq){
    verbose_level_ = 0;
    rad_ = rad;
    radq_ = radq;
    thq_ = thq;
    histq_ = histq;

    desc_ = new daisy();
    desc_->set_parameters(rad_, radq_, thq_, histq_);
}

DaisyDescriptorExtractor::~DaisyDescriptorExtractor(){
    delete desc_;
}

void DaisyDescriptorExtractor::compute_dense(const cv::Mat& image,
                                       cv::Mat& descriptors){
    if (image.channels() != 1){
        CV_Error(CV_StsBadSize, "input image needs to be single-channel");
    }

    desc_->reset();
    desc_->verbose( verbose_level_ ); // 0,1,2,3 -> how much output do you want while running

    // NOTE: big possible bug here, image is of a different type from float

    float * image_mem = new float[image.rows * image.cols];
    Mat image_(image.rows,image.cols,CV_32FC1,image_mem);
    image.convertTo(image_, CV_32F);

    desc_->set_image<float>(image_mem, image_.rows, image_.cols);
    desc_->set_parameters(rad_, radq_, thq_, histq_); // default values are 15,3,8,8

    desc_->initialize_single_descriptor_mode();
    desc_->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED!)

    //desc_->save_descriptors_ascii("/home/dchu/Desktop/test");

    // the descriptors are not normalized yet
    desc_->normalize_descriptors();

    // get access to the dense descriptor
    Mat internal_descriptor(image.rows*image.cols, desc_->descriptor_size(), CV_32F, desc_->get_dense_descriptors());

//    for (int i=0; i<10000; i++)
//        std::cout << internal_descriptor.at<float>(i,0);

    if (descriptors.empty()){
        // return a copy
        descriptors = internal_descriptor.clone();
    } else {
        // we hope that the user has allocated enough space
        // the user will know if the below fails
        internal_descriptor.copyTo(descriptors);
    }

    delete [] image_mem;
}

/** \brief Compute sparse descriptor

  \note No interpolation is performed
*/
void DaisyDescriptorExtractor::compute(const cv::Mat& image,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::Mat& descriptors){
    // Compute dense descriptor
    Mat dense_descriptor;
    compute_dense(image, dense_descriptor);

    // allocate new memory only when necessary
    if (descriptors.empty())
        descriptors = Mat(keypoints.size(), dense_descriptor.cols, dense_descriptor.type());
    else {
        CV_Assert(descriptors.rows == keypoints.size());
        CV_Assert(descriptors.cols == dense_descriptor.cols);
        CV_Assert(descriptors.type() == dense_descriptor.type());
    }

    int x,y;
    // Retrieve only the points we need
    for (int i=0; i<keypoints.size(); i++){
        // no interpolation is performed
        x = (int)keypoints[i].pt.x;
        y = (int)keypoints[i].pt.y;

        Mat current_row = descriptors.row(i);
        Mat target_row = dense_descriptor.row(y*image.cols + x);
        target_row.copyTo(current_row);
    }
}

/** \brief Compute sparse descriptors

  A convenience method so you can just loadtxt the query points and pass in here

  \param image input image (one channel only)
  \param query_points an Nx2 matrix, each row contains the x,y coordinate in the image to query for features
  \param[out] descriptors descriptors to write out (e.g. Nx200 if default constructor of DaisyDescriptorExtractor is used)
  */
void DaisyDescriptorExtractor::compute(const cv::Mat& image,
                                     cv::Mat& query_points,
                                     cv::Mat& descriptors){
    // check that input image is only single channel
    CV_Assert(image.channels() == 1)

    Mat dense_descriptors_im;
    compute_dense(image, dense_descriptors_im);

    // We need to compute_dense first so the DAISY descriptor object knows how long is its descriptor
    // allocate new memory only when necessary
    if (descriptors.empty()){
        descriptors.create(query_points.rows, desc_->descriptor_size(), CV_32F);
    }

    CV_Assert(descriptors.cols == desc_->descriptor_size());
    CV_Assert(descriptors.rows == query_points.rows);
    CV_Assert(descriptors.type() == CV_32F);
    CV_Assert(query_points.type() == CV_32F || query_points.type() == CV_32S)

    // get indexing
    Mat row_indexes = query_points.col(0) + query_points.col(1) * image.cols;

    // get only the features we are interested in
    for (int j=0; j<row_indexes.rows; j++){
        int source_row_index;
        if (query_points.type() == CV_32F){
            source_row_index = (int) row_indexes.at<float>(j,0);
        } else if (query_points.type() == CV_32S){
            source_row_index = (int) row_indexes.at<int>(j,0);
        } else {
            CV_Error(CV_StsUnsupportedFormat, "query_points matrix can only be either CV_32F or CV_32S");
        }

        // NOTE: we need to offset the row because of how we are storing descriptors
        //       We are simply vertical-stack all descriptors together.
        Mat target_row = descriptors.row(j);

        dense_descriptors_im.row(source_row_index).copyTo(target_row);
    }
}

int DaisyDescriptorExtractor::feature_length() const{
    return desc_->descriptor_size();
}
