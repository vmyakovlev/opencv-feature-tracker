#include <cv.h>
#include <fstream>
#include "misc.h"

using std::string;
using std::ifstream;
using std::ofstream;
using std::endl;
using cv::Mat;

// Template specialization to set size to 1 for output KeyPoint
template<>
void vector_one_to_another<cv::Point2f,cv::KeyPoint>(const vector<cv::Point2f> in, vector<cv::KeyPoint> & out){
    out.clear();
    out.resize(in.size());

    for (int i=0; i<in.size(); i++){
        out[i] = cv::KeyPoint(in[i], 1);
    }
}

/** \brief Read a space/tab -separated file.

  The first line will provide the number of columns we are going to read

  \note Only read float now.
  \bug Does not deal with white space at the end the first line.
  \bug Does not deal with comment character (e.g. #)
  \bug Does not deal with column skipping (i.e. not possible to read files that have non-numeric columns)
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