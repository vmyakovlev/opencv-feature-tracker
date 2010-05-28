#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <string>
#include <vector>

#include "nch_feature.h"
#include "distance.h"
#include "splitter.h"
#include "region_distance.h"
#include "global_distance.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace cv;

class MouseParam {
public:
	MouseParam(const Mat& mat, const string& win_name) {
		img = mat;
		window_name = win_name;
		Reset();
	}

	void Reset() {
		upper_left[0] = -1;
		upper_left[1] = -1;
		lower_right[0] = -1;
		lower_right[1] = -1;
	}

	Mat img;
	Vec2i upper_left;
	Vec2i lower_right;
	Mat roi;
	string window_name;
} ;

void Mouse(int event, int x, int y, int flags, void* param) {
	MouseParam* mp = (MouseParam*)param;
	Mat img = mp->img.clone();
	switch (event) {
		case CV_EVENT_LBUTTONDOWN:
			mp->upper_left[0] = x;
			mp->upper_left[1] = y;
			break;
		case CV_EVENT_LBUTTONUP:
			mp->lower_right[0] = x;
			mp->lower_right[1] = y;
			cvRectangle(&IplImage(img),
				cvPoint(mp->upper_left[0], mp->upper_left[1]),
				cvPoint(mp->lower_right[0], mp->lower_right[1]),
				cvScalar(0, 0, 255, 0), 2, 8, 0);
			imshow(mp->window_name, img);
			mp->roi = mp->img(Rect(Point(mp->upper_left), Point(mp->lower_right)));
			break;
		case CV_EVENT_MOUSEMOVE:
			/* draw a rectangle */
			if (mp->upper_left[0] != -1 && mp->upper_left[1] != -1 &&
				mp->lower_right[0] == -1 && mp->lower_right[1] == -1) {
				cvRectangle(&IplImage(img),
					cvPoint(mp->upper_left[0], mp->upper_left[1]),
					cvPoint(x, y),
					cvScalar(0, 0, 255, 0), 2, 8, 0);
				imshow(mp->window_name, img);
			}
			break;
		default:
			break;
	}
}

Mat GetMaskedImage(const Mat& bg, const MouseParam& param, const double& thres = 12.0) {
	Mat bg_roi = bg(Rect(Point(param.upper_left), Point(param.lower_right)));
	//namedWindow("Local Background", CV_WINDOW_AUTOSIZE);
	//imshow("Local Background", bg_roi);

	Mat diff = abs(param.roi - bg_roi);
	Mat diff_rgb[3];
	split(diff, diff_rgb);
	diff = diff_rgb[0] + diff_rgb[1] + diff_rgb[2];
	diff = diff / 3.0;
	//namedWindow("Local Difference", CV_WINDOW_AUTOSIZE);
	//imshow("Local Difference", diff);

	Mat mask(diff.rows, diff.cols, CV_8U);
	// arguable threshold
	threshold(diff, mask, thres, 255, THRESH_BINARY);
	Mat temp;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
	morphologyEx(mask, temp, MORPH_OPEN, kernel);
	morphologyEx(temp, mask, MORPH_CLOSE, kernel);
	//namedWindow("Foreground Mask", CV_WINDOW_AUTOSIZE);
	//imshow("Foreground Mask", mask);

	// prepare masked input, will have value -1 where the mask is 0
	Mat input_roi_rgb[3];
	split(param.roi, input_roi_rgb);
	int rows = mask.rows, cols = mask.cols;
	Mat masked_input(rows, cols, CV_32FC3);
	Mat masked_input_rgb[3];
	split(masked_input, masked_input_rgb);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j) {
			if (mask.at<uchar>(i,j) == 0) {
				masked_input_rgb[0].at<float>(i,j) = -1;
				masked_input_rgb[1].at<float>(i,j) = -1;
				masked_input_rgb[2].at<float>(i,j) = -1;
			}
			else {
				masked_input_rgb[0].at<float>(i,j) = static_cast<float>(input_roi_rgb[0].at<uchar>(i,j));
				masked_input_rgb[1].at<float>(i,j) = static_cast<float>(input_roi_rgb[1].at<uchar>(i,j));
				masked_input_rgb[2].at<float>(i,j) = static_cast<float>(input_roi_rgb[2].at<uchar>(i,j));
			}
		}
	merge(masked_input_rgb, 3, masked_input);
	//namedWindow("Masked Image", CV_WINDOW_AUTOSIZE);
	//imshow("Masked Image", masked_input);

	return masked_input;
}

int main(int argc, char** argv) {

	if (argc < 5) {
		cout << "Usage: " << argv[0] << " input_frame1.jpg background1.jpg input_frame2.jpg background2.jpg" << endl;
		exit(-1);
	}
	string input1_name = argv[1];
	string bg1_name = argv[2];
	string input2_name = argv[3];
	string bg2_name = argv[4];

	// 1. Read the input frames
	Mat input1 = imread(input1_name);
	Mat input2 = imread(input2_name);

	// 2. Read the background images
	Mat bg1 = imread(bg1_name);
	Mat bg2 = imread(bg2_name);

	// 3. Read from user input to get a bounding box
	MouseParam param(input1, "Input Frame");
	namedWindow("Input Frame", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("Input Frame", &Mouse, &param);
	imshow("Input Frame", input1);
	int key;
	while (1) {
		key = waitKey();
		if (key == 13) break;
		if (key == 27) {
			param.Reset();
			imshow("Input Frame", input1);
		}
	}

	// 4. Extract the local region given the bounding box
	namedWindow("Local Region", CV_WINDOW_AUTOSIZE);
	imshow("Local Region", param.roi);
		
	// 5. Subtract the background pixels from the local region
	Mat masked_input1 = GetMaskedImage(bg1, param);
		
	// 6. Compute the Normalized Color Histogram
	Splitter splitter;
	SplitMode mode = NONE;
	vector<Mat> masked_input1_patches;
	splitter.SplitMat(masked_input1, mode, &masked_input1_patches);
	int num_of_patches = masked_input1_patches.size();
	
	NCHDescriptorExtractor descriptor_extractor(16);
	vector<MatND> input1_descrs;
	descriptor_extractor.compute_dense(masked_input1_patches, input1_descrs);
	
	// 7. Locate candidates in target frame
	vector<MouseParam> candidates;
	Mat drawn = input2.clone();
	MouseParam param2(drawn, "Target Frame");
	namedWindow("Target Frame", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("Target Frame", &Mouse, &param2);
	imshow("Target Frame", input2);
	while (1) {
		key = waitKey();
		if (key == 13) {
			if (param2.lower_right[0] != -1 && param2.lower_right[0] != -1)
				candidates.push_back(param2);
			break;
		}
		if (key == 's') {
			// otherwise the ROI will have red line artifacts
			param2.roi = input2(Rect(Point(param2.upper_left), Point(param2.lower_right)));
			candidates.push_back(param2);
			cvRectangle(&IplImage(drawn),
				cvPoint(param2.upper_left[0], param2.upper_left[1]),
				cvPoint(param2.lower_right[0], param2.lower_right[1]),
				cvScalar(0, 0, 255, 0), 2, 8, 0);
			param2.Reset();
		}
		if (key == 27) {
			param2.Reset();
			candidates.clear();
			drawn = input2.clone();
			param2.img = drawn;
			imshow("Target Frame", input2);
		}
	}
	
	Mat descr2_rgb[3];
    RegionDistance<PearsonCoefficientDistance> region_distance;
    GlobalDistance<PearsonCoefficientDistance> global_distance;
	int num_of_candidates = candidates.size();
	for (int i = 0; i < num_of_candidates; ++i) {
		Mat masked_input2 = GetMaskedImage(bg2, candidates[i]);
		vector<Mat> masked_input2_patches;
		vector<MatND> input2_descrs;
		splitter.SplitMat(masked_input2, mode, &masked_input2_patches);
		descriptor_extractor.compute_dense(masked_input2_patches, input2_descrs);
		
		//for (int j = 0; j < num_of_patches; ++j) {
		//	imshow("Masked Image", masked_input2_patches[j]);
		//	waitKey();
		//}
	
		float region_dist = region_distance(input1_descrs, input2_descrs);
		printf("Region distance for Candidate %d : %.3f\n", i+1, region_dist);
		float global_dist = global_distance(input1_descrs, input2_descrs);
		printf("Global distance for Candidate %d : %.3f\n", i+1, global_dist);

		cout << endl;
	}
	waitKey();
	return 0;
}
