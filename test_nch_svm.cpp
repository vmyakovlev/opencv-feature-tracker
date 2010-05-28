#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cv.h>
#include <highgui.h>

#include "nch_feature.h"
#include "svm_classifier.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;
using namespace::cv;

Mat GetMaskedImage(const Mat& img, const Mat& bg, const Rect& roi, const double& thres = 12.0) {
	Mat bg_roi = bg(roi);
	//namedWindow("Local Background", CV_WINDOW_AUTOSIZE);
	//imshow("Local Background", bg_roi);

	Mat diff = abs(img - bg_roi);
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
	split(img, input_roi_rgb);
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

#if 0 // for testing the reshape function, if possible should be moved to TEST
	int size[3] = {2,2,2};
	MatND foo(3, size, CV_32F);
	foo.at<float>(0,0,0) = 1;
	foo.at<float>(0,0,1) = 2;
	foo.at<float>(1,0,0) = 3;
	foo.at<float>(1,0,1) = 4;
	foo.at<float>(0,1,0) = 5;
	foo.at<float>(0,1,1) = 6;
	foo.at<float>(1,1,0) = 7;
	foo.at<float>(1,1,1) = 8;
	Mat bar(1, 8, CV_32F);
	bar = Scalar(0);
	for (int i = 0; i < 8; ++i) {
		// according to MATLAB's reshape rule, column by column and then slice by slice
		int slice = i / 4;
		int col = (i - slice * 4) / 2;
		int row = (i - slice * 4) % 2;
		bar.at<float>(0, i) = foo.at<float>(row, col, slice);
		cout << bar.at<float>(0,i) << " ";
	}
	cout << endl;
#endif

	const string data_dir = "C://workspace//My Dropbox//Shared//ACV-HW2//Data//Reacquisition//Testing//";
	const string bg_dir = "C://workspace//My Dropbox//Shared//ACV-HW2//background//";
	namedWindow("Current Frame", CV_WINDOW_AUTOSIZE);

	if (argc < 3) {
		cout << "Usage: " << argv[0] << " GROUNDTRUTH_FILE MODEL_LIST" << endl;
		exit(-1);
	}

	string gt_file = argv[1];
	int pos1 = gt_file.find_first_of('_');
	int pos2 = gt_file.find_last_of('_');
	string view = gt_file.substr(pos1+1, pos2-pos1-1);
	string view_dir = data_dir + "View_" + view + "//";
	string bg_file = bg_dir + "bg" + view + ".png";

	/** reading the model list file, KEEP THE FILENAMES SEQUENTIAL!!! */
	string model_list = argv[2];
	ifstream model_in;
	model_in.open(model_list.c_str(), ifstream::in);
	if (model_in.fail()) {
		cout << "Could not open " << model_list << " for reading model filenames." << endl;
		exit(-1);
	}
	vector<string> filenames;
	string filename;
	while (model_in >> filename) {
		filenames.push_back(filename);
	}
	model_in.close();
	
	ifstream fin;
	fin.open(gt_file.c_str(), ifstream::in);
	if (fin.fail()) {
		cout << "Could not open " << gt_file << " for reading groundtruth." << endl;
		exit(-1);
	}

	NCHDescriptorExtractor descriptor_extractor;
	Mat bg = imread(bg_file);
	//imshow("Current Frame", bg);
	//waitKey();

	/** get some dimensions */
	int bins = descriptor_extractor.get_bins();
	int slice_size = bins * bins;
	int cols = bins * bins * bins;

	/** construct the SVM models */
	SVMClassifier classifier(cols);
	if (!classifier.LoadModels(filenames))
		exit(-1);

	int fr_id, subj_id, ctr = 0, correct = 0;
	float xf, yf, wf, hf;
	char str[256];
	while (fin >> fr_id) {
		
		fin >> subj_id >> xf >> yf >> wf >> hf;
		if (subj_id == 0)
			continue;
		
		sprintf(str, "frame_%04d.jpg", fr_id);
		string frame_file = view_dir + str;
		
		Mat frame = imread(frame_file);
		imshow("Current Frame", frame);
		cout << "Processing " << str;

		int x = static_cast<int>(ceil(xf + 0.5)); x = x < 0 ? 0 : x;
		int y = static_cast<int>(ceil(yf + 0.5)); y = y < 0 ? 0 : y;
		int w = static_cast<int>(ceil(wf + 0.5)); w = (x+w) > frame.cols ? frame.cols - x : w;
		int h = static_cast<int>(ceil(hf + 0.5)); h = (y+h) > frame.rows ? frame.rows - y : h;
		Rect roi(x, y, w, h);
		Mat frame_roi = frame(roi);
		Mat frame_roi_masked = GetMaskedImage(frame_roi, bg, roi);
		MatND descriptor;
		descriptor_extractor.compute_dense(frame_roi_masked, descriptor);
				
		Mat descriptor_1d(1, cols, CV_32F);
		descriptor_1d = Scalar(0);
		for (int i = 0; i < cols; ++i) {
			// according to MATLAB's reshape rule, column by column and then slice by slice
			int slice = i / slice_size;
			int col = (i - slice * slice_size) / bins;
			int row = (i - slice * slice_size) % bins;
			descriptor_1d.at<float>(0, i) = descriptor.at<float>(row, col, slice);
		}

 		int predicted_id = classifier.Classify(descriptor_1d);
		cout << "\tReal: " << subj_id << "\tPredict: " << predicted_id << endl;
		if (subj_id == predicted_id) correct++;

		waitKey(1);
		ctr++;
	}
	fin.close();
	
	float identification = 1.0f * correct / ctr;
	cout << "Identification Rate: " << identification << endl;

	return 0;
}

