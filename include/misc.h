#ifndef __MISC_H_
#define __MISC_H_
#include <vector>
#include <cv.h>
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

cv::Mat loadtxt(std::string filename);
bool WriteTXT(const std::string& filename, const cv::Mat& mat);
void LoadGalleryPCAs(std::vector<cv::PCA> * gallery_PCAs, const char * pca_folder, int num_gallery_subjects);
cv::Mat FindInterestPoints(const cv::Mat & gray_im, const char * query_point_filename, int max_num_corners = 10);
#endif

