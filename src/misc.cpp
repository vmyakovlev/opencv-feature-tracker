#include <cv.h>
#include <iostream>
#include <fstream>
#include "opencv_io_extra.h"
#include "misc.h"

using namespace cv;

using std::string;
using std::ifstream;
using std::ofstream;
using std::endl;
using std::cout;
using std::vector;

// Template specialization to set size to 1 for output KeyPoint
template<>
void vector_one_to_another<cv::Point2f,cv::KeyPoint>(const vector<cv::Point2f> in, vector<cv::KeyPoint> & out){
    out.clear();
    out.resize(in.size());

    for (int i=0; i<in.size(); i++){
        out[i] = cv::KeyPoint(in[i], 1);
    }
}

// Template specialization to get the points from KeyPoint
template<>
void vector_one_to_another<cv::KeyPoint,cv::Point2f>(const vector<cv::KeyPoint> in, vector<cv::Point2f> & out){
    out.clear();
    out.resize(in.size());

    for (int i=0; i<in.size(); i++){
        out[i] = in[i].pt;
    }
}

/** \brief Read a space/tab -separated file.

  The first line will provide the number of columns we are going to read

  \note Only read float now.
  \todo Templatize to read other datatypes
  \bug Does not deal with white space at the end the lines.
  \bug Does not deal with comment character (e.g. #)
  \bug Does not deal with column skipping (i.e. not possible to read files that have non-numeric columns)
  \bug Does not deal with blank lines
*/
Mat loadtxt(string filename){
    ifstream ifs(filename.c_str());

    if (!ifs.is_open()){
        string error_string = "Cannot open file " + filename;
        throw cv::Exception(CV_StsBadArg, error_string, "loadtxt", "misc.cpp", 30);
    }

    // we read the first line to figure out the number of column
    string first_line;
    int number_of_columns = 1;
    getline(ifs, first_line);

    // figure out how many columns there are
    size_t found;

    found=first_line.find_first_of(" \t");
    while (found!=string::npos)
    {
        number_of_columns++;
        found=first_line.find_first_of(" \t",found+1);
    }

    // start reading from the begining
    ifs.seekg(0);
    vector<float> data;
    float datum;
    ifs >> datum;
    while(!ifs.eof()){
        data.push_back(datum);
        ifs >> datum;
    }

    // convert our vector to the matrix
    CV_Assert(data.size() % number_of_columns == 0);
    int number_of_rows = data.size() / number_of_columns;

    Mat loaded_mat(number_of_rows, number_of_columns, CV_32F);
    int count = 0;
    for (int i=0; i<number_of_rows; i++){
        for (int j=0; j<number_of_columns; j++){
            loaded_mat.at<float>(i,j) = data[count];
            count++;
        }
    }

    return loaded_mat;
}

/** \brief Writes a space/tab -separated file.

  \note Only writes single channel float now.
*/
bool WriteTXT( const std::string& filename, const cv::Mat& mat )
{
	CV_Assert(mat.type() == CV_32F);

	ofstream ofs(filename.c_str(), ofstream::out);

	if (!ofs.is_open()){
		string error_string = "Cannot open file " + filename;
		return false;
	}

	cv::Mat_<float> ez_mat = mat;
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			ofs << ez_mat[i][j] << "\t";
		}
		ofs << endl;
	}

	ofs.close();

	return true;
}

/** \brief Load gallery PCAs from a specific folder

  The file has to be of the following pattern: subject%d_pca.yml (%d => subject id e.g. 1,2,3,4, ...)
*/
void LoadGalleryPCAs(vector<PCA> * gallery_PCAs, const char * pca_folder, int num_gallery_subjects){
    gallery_PCAs->resize(num_gallery_subjects);
    char pca_filename[256];

    // NOTE: subject naming starts at 1 (MATLAB-like)
    for (int i=1; i<=num_gallery_subjects; i++){
        // load individual PCA
        sprintf(pca_filename, "%s/subject%d_pca.yml", pca_folder, i);
        std::cout << "Reading PCA saved in " << pca_filename << std::endl;
        FileStorage fs(string(pca_filename), FileStorage::READ);
        read(fs, (*gallery_PCAs)[i-1]);
    }
}

/** \brief Find the interest points within the bounding box given by the query_point_filename
*/
Mat FindInterestPoints(const Mat & gray_im, const char * query_point_filename, int max_num_corners /*= 10*/){
    Mat query_points = loadtxt(query_point_filename);

    // the top left corner is assumed to be the first entry
    Point2i top_left((int) query_points.at<float>(0,0), (int) query_points.at<float>(0,1));
    int last_row_id = query_points.rows - 1;
    Point2i bottom_right((int) query_points.at<float>(last_row_id, 0), (int) query_points.at<float>(last_row_id, 1));

    // insanity check on these points
    if (top_left.x < 0)
        top_left.x = 0;
    if (top_left.y < 0)
        top_left.y = 0;
    if (bottom_right.x > gray_im.cols-1)
        bottom_right.x = gray_im.cols-1;
    if (bottom_right.y > gray_im.rows-1)
        bottom_right.y = gray_im.cols-1;

    // the bottom right corner is assumed to be the last entry
    Rect roi_rect(top_left, bottom_right);

    // get the image roi
    Mat roi(gray_im, roi_rect);

    // Find corner points
    vector<Point2f> corners;
    goodFeaturesToTrack(roi, corners, max_num_corners, 0.1, 10);

    // Return the query points
    Mat interest_points(corners.size(), 2, CV_32S);
    for (int i=0; i<corners.size(); ++i){
        // Translate these coordinates back into original image coordinates
        interest_points.at<int>(i,0) = (int)corners[i].x + top_left.x;
        interest_points.at<int>(i,1) = (int)corners[i].y + top_left.y;

        //std::cout << interest_points.at<int>(i,0) << " " << interest_points.at<int>(i,1) << std::endl;

    }

    return interest_points;
}

std::ostream& operator<< (std::ostream& out, const cv::Vec2f & vec )
{
    out << vec[0] << " " << vec[1] << " ";
    return out;
};
