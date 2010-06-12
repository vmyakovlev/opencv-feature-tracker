#ifndef __MISC_H_
#define __MISC_H_
#include <vector>
#include <cv.h>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;

/** \brief Create an array of array from a vector of vector

Make sure you free the data with delete_arr_arr.
Notice that T2 needs to be convertible to T

\see delete_arr_arr
*/
template <class T, class T2>
T** vec_vec_to_arr_arr(vector<vector <T2> > input_vector){
    T** result = new T*[input_vector.size()];

    for (int i = 0; i < input_vector.size() ; i++)
    {
        result[i] = new T[input_vector[i].size()];
        for (int j = 0; j < input_vector[i].size() ; j++)
        {
            // Let's do the copying
            result[i][j] = input_vector[i][j];
        }
    }
    return result;
}

/** \brief Deallocate the memory of an array of array
*/
template <class T>
void delete_arr_arr(T** arr, int size){
    for (int i=0; i<size; i++)
        delete [] arr[i];

    delete [] arr;
}

/** \brief Convert one vector of T to another vector T2

  There needs to be T(T2 input) function
*/
template<class T, class T2>
void vector_one_to_another(const vector<T > in, vector<T2> & out){
    out.clear();
    out.resize(in.size());

    for (int i=0; i<in.size(); i++)
        out[i] = in[i];
}

template<>
void vector_one_to_another<cv::Point2f,cv::KeyPoint>(const vector<cv::Point2f> in, vector<cv::KeyPoint> & out);
template<>
void vector_one_to_another<cv::KeyPoint,cv::Point2f>(const vector<cv::KeyPoint> in, vector<cv::Point2f> & out);

cv::Mat loadtxt(std::string filename);
bool WriteTXT(const std::string& filename, const cv::Mat& mat);
void LoadGalleryPCAs(std::vector<cv::PCA> * gallery_PCAs, const char * pca_folder, int num_gallery_subjects);
cv::Mat FindInterestPoints(const cv::Mat & gray_im, const char * query_point_filename, int max_num_corners = 10);

/** \brief Print the content of the matrix

  \tparam The type of matrix elements
*/
template<typename T> void print_matrix(const cv::Mat & mat){
    for (int i=0; i<mat.rows; i++){
        const T* mat_ptr = mat.ptr<T>(i);
        std::cout << ">   ";
        for (int j=0; j<mat.cols; j++){
            std::cout << mat_ptr[j] << " ";
        }

        std::cout << std::endl;
    }
}

/** Indexing into the container given the indices

  e.g. a = [4,1,2,5,3]
  new_indices = [2,3,4,1,0]

  This method does similar to a[new_indices]. That is it returns:
    [2,5,3,1,4]
*/
template<typename VT>
VT indexing(VT input, const vector<int> & indices){
    VT output(input.size());

    for (int i=0; i<indices.size(); i++){
        output[i] = input[indices[i]];
    }

    return output;
}

// Providing printing support for Vec2f
std::ostream& operator<< (std::ostream& out, const cv::Vec2f & vec );

// Conversion between world and image coordinate
// NOTE: You can pass just use one and manually invert the homography_matrix
void convert_to_world_coordinate(const vector<cv::Point2f> & points_in_image_coordinate,
                                 const cv::Mat & homography_matrix,
                                 vector<cv::Point2f> * points_in_world_coordinate);
void convert_to_image_coordinate(const cv::Point2f & point_in_world_coordinate, const cv::Mat & homography_matrix, cv::Point2f * point_in_image_coordinate);
void convert_to_image_coordinate(const vector<cv::Point2f> & points_in_world_coordinate,
                                 const cv::Mat & homography_matrix,
                                 vector<cv::Point2f> * points_in_image_coordinate);
#endif

