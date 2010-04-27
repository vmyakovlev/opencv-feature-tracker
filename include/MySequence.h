#pragma once

#include <iostream>
#include <list>
#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

struct Frames
{
		Mat frame_view_005, frame_view_007, frame_view_008;
		Mat sub_view_005, sub_view_007, sub_view_008; 
};

struct Ground_truth // has info per subject
{
		int frame_no;
		int sub_id;
		
		double top_left_x_005, top_left_y_005, width_005, height_005;
		double top_left_x_007, top_left_y_007, width_007, height_007;
		double top_left_x_008, top_left_y_008, width_008, height_008;

		double center_x_005, center_y_005; //these can be calculated from the rest
		double center_x_007, center_y_007;
		double center_x_008, center_y_008;

		Mat view_005_ROI;
		Mat view_007_ROI;
		Mat view_008_ROI;

		Rect view_005_rect;
		Rect view_007_rect;
		Rect view_008_rect;

		bool isPresent_005, isPresent_007, isPresent_008;

		//void initGroundTruth(void);

};

struct IDList
{
	list<int> ids_005;
	list<int> ids_007;
	list<int> ids_008;
};

class MySequence
{
private:
	
	string path, view_005, view_007, view_008;

	//Frames frame_list;

	Ground_truth g_truth; //current subject data only

	list<Ground_truth> ground_truth_list;

	FILE *file_005, *file_007, *file_008; //, *file_all;

	bool useTruthFlag;

	IDList all_ids;

	//ROIs all_ROIs;
	
	static string getFrameNoString(int);
	void initGroundTruth(void); // set struct Ground_truth to zero
	void initSubFrames(void); // also done in setFramesParallel()
	//void setSubFrames(int,Rect,Rect,Rect); // call per subject with his (frame_no,view_005_rect,view_007_rect,view_008_rect)
	void setSubFrames(int, list<Ground_truth>::iterator);

protected:
	Frames frame_list;
	
public:
	MySequence(void);
	MySequence(string); // (path)
	~MySequence(void);
	void setPathParallel(void); // same as default constructor
	void setPathParallel(string); //same as 2nd constructor
	void setFramesParallel(int); //(frame_no)
	void setFramesSingle(int,int); //(frame_no, view_no)
	void setGroundTruth(int); //(frame_no)
	void setGroundTruthFlag(bool);
	bool getGroundTruthFlag(void);
	Mat getFramesSingle(int, int); //(frame_no, view_no)
	Frames getFramesParallel(void); //(frame_no)
	list<Ground_truth> getGroundTruth(void);
	IDList getIDList(void);
	Size getImageSize(void);
	void populateIDList(void);
	void clearIDList(void);
	void useGroundTruth(int); // (frame_no)
	void clearGroundTruth(int); // (frame_no)
	void showFrames(int,int); //(frame_no,view_no)
	static void showImage(string,Mat);


};
