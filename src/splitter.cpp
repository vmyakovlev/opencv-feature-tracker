#include "splitter.h"
#include <iostream>

using cv::Mat;
using std::cout;
using std::endl;

void Splitter::SplitMat(const cv::Mat& mat_in, SplitMode mode, std::vector<cv::Mat>* mats_out) {
	Mat patch;
	int rows = mat_in.rows;
	int split_idx;
	switch (mode) {
		case NONE:
			//cout << "NONE: Not performing splitting." << endl;
			mats_out->push_back(mat_in);
			break;
		case UL:
			//cout << "UL: Performing top/bottom split." << endl;
			split_idx = floor(rows / 2.0f + 0.5f);
			patch = mat_in(cv::Range(0,split_idx), cv::Range::all());
			mats_out->push_back(patch);
			patch = mat_in(cv::Range(split_idx,rows), cv::Range::all());
			mats_out->push_back(patch);
			break;
		case ULF:
			cout << "ULF: Performing top/bottom/foot split." << endl;
			cout << "But too bad, this has not been implemented yet." << endl;
			break;
		default:
			cout << "UNKNOWN: Invalid split mode." << endl;
			break;
	}
};