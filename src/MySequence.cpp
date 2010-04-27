#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <list>
#include "cv.h"
#include "highgui.h"
#include "MySequence.h"

using namespace std;
using namespace cv;

MySequence::MySequence(void)
{
	this->path = "C:\\Get In\\Aabhyas Spring 2010\\Advanced Computer Vision\\Assignments\\Assignment 2\\Reacquisition\\Reacquisition\\Testing";
	this->view_005 = path + "\\View_005";
	this->view_007 = path + "\\View_007";
	this->view_008 = path + "\\View_008";
	
	this->file_005 = fopen("view005_box.txt","r");
	if(this->file_005 == NULL)
	{
		cout<<"Error opening file view005_box.txt"<<endl;
		exit(1);
	}
	
	this->file_007 = fopen("view007_box.txt","r");
	if(this->file_007 == NULL)
	{
		cout<<"Error opening file view007_box.txt"<<endl;
		exit(2);
	}

	this->file_008 = fopen("view008_box.txt","r");
	if(this->file_008 == NULL)
	{
		cout<<"Error opening file view008_box.txt"<<endl;
		exit(3);
	}

	this->initGroundTruth();
	this->useTruthFlag = false;

	
}

MySequence::MySequence(string path)
{
	this->path = path;
	this->view_005 = path + "\\View_005";
	this->view_007 = path + "\\View_007";
	this->view_008 = path + "\\View_008";

	this->file_005 = fopen("view005_box.txt","r");
	if(this->file_005 == NULL)
	{
		cout<<"Error opening file view005_box.txt"<<endl;
		exit(1);
	}
	
	this->file_007 = fopen("view007_box.txt","r");
	if(this->file_007 == NULL)
	{
		cout<<"Error opening file view007_box.txt"<<endl;
		exit(2);
	}

	this->file_008 = fopen("view008_box.txt","r");
	if(this->file_008 == NULL)
	{
		cout<<"Error opening file view008_box.txt"<<endl;
		exit(3);
	}

	this->initGroundTruth();
	this->useTruthFlag = false;

	
}

MySequence::~MySequence(void)
{
}

void MySequence::setPathParallel()
{
	this->path = "C:\\Get In\\Aabhyas Spring 2010\\Advanced Computer Vision\\Assignments\\Assignment 2\\Reacquisition\\Reacquisition\\Testing";
	this->view_005 = path + "\\View_005";
	this->view_007 = path + "\\View_007";
	this->view_008 = path + "\\View_008";

}
void MySequence::setPathParallel(string path)
{
	this->path = path;
	this->view_005 = path + "\\View_005";
	this->view_007 = path + "\\View_007";
	this->view_008 = path + "\\View_008";
}

void MySequence::setFramesParallel(int count)
{
	string frame_no = getFrameNoString(count);

	this->frame_list.frame_view_005 = imread(this->view_005 + frame_no,1);
	this->frame_list.frame_view_007 = imread(this->view_007 + frame_no,1);
	this->frame_list.frame_view_008 = imread(this->view_008 + frame_no,1);

	//cout<<view_005+frame_no;

	//showImage("view_005" ,this->frame_list.frame_view_005);
	//showImage("view_007" ,this->frame_list.frame_view_007);
	//showImage("view_008" ,this->frame_list.frame_view_008);

	this->frame_list.sub_view_005 = this->frame_list.frame_view_005.clone();
	this->frame_list.sub_view_007 = this->frame_list.frame_view_007.clone();
	this->frame_list.sub_view_008 = this->frame_list.frame_view_008.clone();

	this->frame_list.sub_view_005.setTo(Scalar(0,0,0),Mat());
	this->frame_list.sub_view_007.setTo(Scalar(0,0,0),Mat());
	this->frame_list.sub_view_008.setTo(Scalar(0,0,0),Mat());
		
}

void MySequence::setFramesSingle(int count, int view_no)
{

	string frame_no = getFrameNoString(count); 

	if(view_no == 5)
	{
		this->frame_list.frame_view_005 = imread(this->view_005 + frame_no,1);
		//showImage("view_005" ,this->frame_list.frame_view_005);
	}
	else if(view_no == 7)
	{
		this->frame_list.frame_view_007 = imread(this->view_007 + frame_no,1);
		//showImage("view_007" ,this->frame_list.frame_view_007);
	}
	else if(view_no == 8)
	{
		this->frame_list.frame_view_008 = imread(this->view_008 + frame_no,1);
		//showImage("view_008" ,this->frame_list.frame_view_008);
	}
	else
	{
		cout<<"Invalid view number!"<<endl;
		return;
	}
	
}

Mat MySequence::getFramesSingle(int count, int view_no)
{
	string frame_no = getFrameNoString(count);
	
	if(view_no == 5)
		return this->frame_list.frame_view_005;
	else if(view_no == 7)
		return this->frame_list.frame_view_007;
	else if(view_no == 8)
		return this->frame_list.frame_view_008;
	else
	{	
		cout<<"Invalid view number!"<<endl;
		//return;
	}

}

Frames MySequence::getFramesParallel(void)
{
	return this->frame_list;
}

string MySequence::getFrameNoString(int count)
{
	stringstream ss;
	ss<<count;
	string curr = ss.str(); 
	//cout<<curr<<endl; 

	string frame_no;
	if(count<10)
		frame_no = "\\frame_000" + curr + ".jpg";
	else if(count<100)
		frame_no = "\\frame_00" + curr + ".jpg";
	else if(count<1000)
		frame_no = "\\frame_0" + curr + ".jpg";

	return frame_no;
}

void MySequence::initGroundTruth(void)
{
	this->g_truth.frame_no = 0;
	this->g_truth.sub_id = 0;
	this->g_truth.center_x_005 = 0.0;
	this->g_truth.center_x_007 = 0.0;
	this->g_truth.center_x_008 = 0.0;
	this->g_truth.center_y_005 = 0.0;
	this->g_truth.center_y_007 = 0.0;
	this->g_truth.center_y_008 = 0.0;
	this->g_truth.height_005 = 0.0;
	this->g_truth.height_007 = 0.0;
	this->g_truth.height_008 = 0.0;
	this->g_truth.top_left_x_005 = 0.0;
	this->g_truth.top_left_x_007 = 0.0;
	this->g_truth.top_left_x_008 = 0.0;
	this->g_truth.top_left_y_005 = 0.0;
	this->g_truth.top_left_y_007 = 0.0;
	this->g_truth.top_left_y_008 = 0.0;
	this->g_truth.width_005 = 0.0;
	this->g_truth.width_007 = 0.0;
	this->g_truth.width_008 = 0.0;

	Rect r_temp(0,0,0,0);
	this->g_truth.view_005_rect = r_temp;
	this->g_truth.view_007_rect = r_temp;
	this->g_truth.view_008_rect = r_temp;

	this->g_truth.view_005_ROI.create(this->frame_list.frame_view_005.size(),this->frame_list.frame_view_005.type());// = this->frame_list.frame_view_005.clone();//NULL;// setTo(Scalar(0,0,0),Mat());
	this->g_truth.view_007_ROI.create(this->frame_list.frame_view_007.size(),this->frame_list.frame_view_007.type());// = this->frame_list.frame_view_007.clone();//NULL;// setTo(Scalar(0,0,0),Mat());
	this->g_truth.view_008_ROI.create(this->frame_list.frame_view_008.size(),this->frame_list.frame_view_008.type());// = this->frame_list.frame_view_008.clone();//NULL;// setTo(Scalar(0,0,0),Mat());
	
	//this->g_truth.view_005_ROI.setTo(Scalar(0,0,0),Mat());
	//showImage("init roi005",this->g_truth.view_005_ROI);
	//waitKey(0);

	this->g_truth.isPresent_005 = false;
	this->g_truth.isPresent_007 = false;
	this->g_truth.isPresent_008 = false;
}

void MySequence::initSubFrames(void)
{
	this->frame_list.sub_view_005 = this->frame_list.frame_view_005.clone();
	this->frame_list.sub_view_007 = this->frame_list.frame_view_007.clone();
	this->frame_list.sub_view_008 = this->frame_list.frame_view_008.clone();

	this->frame_list.sub_view_005.setTo(Scalar(0,0,0),Mat());
	this->frame_list.sub_view_007.setTo(Scalar(0,0,0),Mat());
	this->frame_list.sub_view_008.setTo(Scalar(0,0,0),Mat());
}
void MySequence::setGroundTruth(int count)
{
		
	list<Ground_truth>::iterator itr;
	int frame_no=0, sub_id=0, local_frame_count=-1;

	char top_left_x[20],top_left_y[20],width[20], height[20];
	double top_left_x_d,top_left_y_d,width_d, height_d;

	bool flag = false;
	this->file_005 = fopen("view005_box.txt","r");
	this->file_007 = fopen("view007_box.txt","r");
	this->file_008 = fopen("view008_box.txt","r");

	while(!feof(this->file_005))
	{
		fscanf(this->file_005,"%d %d %s %s %s %s ",&frame_no, &sub_id, top_left_x,top_left_y,width,height);
		//cout<<frame_no<<sub_id<<top_left_x<<top_left_y<<width<<height<<endl;
		
		if(frame_no == count)
		{
			top_left_x_d = atof(top_left_x);
			top_left_y_d = atof(top_left_y);
			width_d = atof(width);
			height_d = atof(height);
			
			local_frame_count++;

			this->initGroundTruth();
			
			this->g_truth.frame_no = frame_no;
			this->g_truth.sub_id = sub_id;
			this->g_truth.top_left_x_005 = top_left_x_d;
			this->g_truth.top_left_y_005 = top_left_y_d;
			this->g_truth.width_005 = width_d;
			this->g_truth.height_005 = height_d;
			this->ground_truth_list.push_front(this->g_truth);
		}

					
	}

	while(!feof(this->file_007))
	{
		fscanf(this->file_007,"%d %d %s %s %s %s ",&frame_no, &sub_id, top_left_x,top_left_y,width,height);
		//cout<<frame_no<<sub_id<<top_left_x<<top_left_y<<width<<height<<endl;
		
		if(frame_no == count)
		{
			flag = true;
			
			top_left_x_d = atof(top_left_x);
			top_left_y_d = atof(top_left_y);
			width_d = atof(width);
			height_d = atof(height);
			
			local_frame_count++;
			
			for(itr = this->ground_truth_list.begin(); itr != this->ground_truth_list.end(); itr++)
			{
				if(itr->sub_id == sub_id)
				{
					itr->top_left_x_007 = top_left_x_d;
					itr->top_left_y_007 = top_left_y_d;
					itr->width_007 = width_d;
					itr->height_007 = height_d;

					flag = false;
					break;
				}
			}

			if(flag == true)
			{
				this->initGroundTruth();

				this->g_truth.frame_no = frame_no;
				this->g_truth.sub_id = sub_id;
				this->g_truth.top_left_x_007 = top_left_x_d;
				this->g_truth.top_left_y_007 = top_left_y_d;
				this->g_truth.width_007 = width_d;
				this->g_truth.height_007 = height_d;

				this->ground_truth_list.push_front(this->g_truth);

				//flag = false;
			}
		}

					
	}

	while(!feof(this->file_008))
	{
		fscanf(this->file_008,"%d %d %s %s %s %s ",&frame_no, &sub_id, top_left_x,top_left_y,width,height);
		//cout<<frame_no<<sub_id<<top_left_x<<top_left_y<<width<<height<<endl;
		
		if(frame_no == count)
		{
			flag = true;
			
			top_left_x_d = atof(top_left_x);
			top_left_y_d = atof(top_left_y);
			width_d = atof(width);
			height_d = atof(height);
			
			local_frame_count++;
			
			for(itr = this->ground_truth_list.begin(); itr != this->ground_truth_list.end(); itr++)
			{
				if(itr->sub_id == sub_id)
				{
					itr->top_left_x_008 = top_left_x_d;
					itr->top_left_y_008 = top_left_y_d;
					itr->width_008 = width_d;
					itr->height_008 = height_d;

					flag = false;
					break;
				}
			}

			if(flag == true)
			{
				this->initGroundTruth();

				this->g_truth.frame_no = frame_no;
				this->g_truth.sub_id = sub_id;
				this->g_truth.top_left_x_008 = top_left_x_d;
				this->g_truth.top_left_y_008 = top_left_y_d;
				this->g_truth.width_008 = width_d;
				this->g_truth.height_008 = height_d;

				this->ground_truth_list.push_front(this->g_truth);

				//flag = false;
			}
		}

					
	}
	cout<<endl;

	//Adding ROIs and centers to ground truth

	//list<Ground_truth>::iterator itr2;

	//Rect roi_005, roi_007, roi_008;

	for(itr = this->ground_truth_list.begin(); itr != this->ground_truth_list.end(); itr++)
	{
		itr->center_x_005 = itr->top_left_x_005 + itr->width_005/2;
		itr->center_y_005 = itr->top_left_y_005 + itr->height_005/2;
		itr->center_x_007 = itr->top_left_x_007 + itr->width_007/2;
		itr->center_y_007 = itr->top_left_y_007 + itr->height_007/2;
		itr->center_x_008 = itr->top_left_x_008 + itr->width_008/2;
		itr->center_y_008 = itr->top_left_y_008 + itr->height_008/2;

		if(itr->center_x_005 != 0)  //&& itr->center_y_005 != 0);
			itr->isPresent_005 = true;

		if(itr->center_x_007 != 0)  //&& itr->center_y_007 != 0);
			itr->isPresent_007 = true;

		if(itr->center_x_008 != 0)  //&& itr->center_y_008 != 0);
			itr->isPresent_008 = true;

		Rect roi_005(itr->top_left_x_005,itr->top_left_y_005,itr->width_005,itr->height_005);
		itr->view_005_rect = roi_005;

		Rect roi_007(itr->top_left_x_007,itr->top_left_y_007,itr->width_007,itr->height_007);
		itr->view_007_rect = roi_007;

		Rect roi_008(itr->top_left_x_008,itr->top_left_y_008,itr->width_008,itr->height_008);
		itr->view_008_rect = roi_008;
		
		//showImage("view008_roi",this->frame_list.frame_view_008(roi_008));
		//waitKey(0);

		if(itr->isPresent_005 == true)
			itr->view_005_ROI = this->frame_list.frame_view_005(roi_005).clone();
		if(itr->isPresent_007 == true)
			itr->view_007_ROI = this->frame_list.frame_view_007(roi_007).clone();
		if(itr->isPresent_008 == true)
			itr->view_008_ROI = this->frame_list.frame_view_008(roi_008).clone();

	}

	fclose(this->file_005);
	fclose(this->file_007);
	fclose(this->file_008);

	//this->ground_truth_list.clear();

	

}

void MySequence::showImage(string win_name,Mat img)
{
	namedWindow(win_name,CV_WINDOW_AUTOSIZE);
	imshow(win_name, img);
}

list<Ground_truth> MySequence::getGroundTruth(void)
{
	return this->ground_truth_list;
}
void MySequence::useGroundTruth(int count)
{
	list<Ground_truth> curr_true_list = this->getGroundTruth();
	list<Ground_truth>::iterator itr;

	//initSubFrames();

	for(itr = curr_true_list.begin(); itr != curr_true_list.end(); itr++)
	{
		//setSubFrames(count,itr->view_005_rect,itr->view_007_rect,itr->view_008_rect);
		setSubFrames(count,itr);
	}

	this->setGroundTruthFlag(true);
	
	this->populateIDList();
}
void MySequence::clearGroundTruth(int count)
{
	this->ground_truth_list.clear();
}

void MySequence::setGroundTruthFlag(bool val)
{
	this->useTruthFlag = val;
}

bool MySequence::getGroundTruthFlag(void)
{
	return this->useTruthFlag;
}

void MySequence::setSubFrames(int count, list<Ground_truth>::iterator itr)
{
	Frames curr_frames = this->getFramesParallel();

	//list<Ground_truth> curr_list= this->getGroundTruth(count);
	Rect r_005, r_007, r_008;

	Mat temp_005 = curr_frames.frame_view_005.clone(); 
	uchar* data_005 = temp_005.data;
	int ch_005 = temp_005.channels();

	Mat temp_007 = curr_frames.frame_view_007.clone(); 
	uchar* data_007 = temp_007.data;
	int ch_007 = temp_007.channels();

	Mat temp_008 = curr_frames.frame_view_008.clone(); 
	uchar* data_008 = temp_008.data;
	int ch_008 = temp_008.channels();

	uchar temp;	
	if(itr->isPresent_005 == true)
	{
		for(int y = itr->view_005_rect.y; y < itr->view_005_rect.y + itr->view_005_rect.height; y++)
		{
			for(int x = itr->view_005_rect.x; x < itr->view_005_rect.x + itr->view_005_rect.width; x++)
			{
				for(int k = 0; k < ch_005; k++)
				{
					//this->frame_list.sub_view_005.data[y*temp_005.step + x*ch_005 + k] = 
						temp = data_005[y*temp_005.step + x*ch_005 + k];
						this->frame_list.sub_view_005.data[y*temp_005.step + x*ch_005 + k] =  temp;
				}
			}
		}
	}

	if(itr->isPresent_007 == true)
	{
		//Mat ex_roi = curr_frames.frame_view_007(itr->view_007_rect);
		//showImage("Subject Image",ex_roi);

		for(int y = itr->view_007_rect.y; y < itr->view_007_rect.y + itr->view_007_rect.height; y++)
		{
			for(int x = itr->view_007_rect.x; x < itr->view_007_rect.x + itr->view_007_rect.width; x++)
			{
				for(int k = 0; k < ch_007; k++)
				{
					//this->frame_list.sub_view_005.data[y*temp_005.step + x*ch_005 + k] = 
						temp = data_007[y*temp_007.step + x*ch_007 + k];
						this->frame_list.sub_view_007.data[y*temp_007.step + x*ch_007 + k] =  temp;
				}
			}
		}
	}

	if(itr->isPresent_008 == true)
	{
		for(int y = itr->view_008_rect.y; y < itr->view_008_rect.y + itr->view_008_rect.height; y++)
		{
			for(int x = itr->view_008_rect.x; x < itr->view_008_rect.x + itr->view_008_rect.width; x++)
			{
				for(int k = 0; k < ch_008; k++)
				{
					//this->frame_list.sub_view_005.data[y*temp_005.step + x*ch_005 + k] = 
						temp = data_008[y*temp_008.step + x*ch_008 + k];
						this->frame_list.sub_view_008.data[y*temp_008.step + x*ch_008 + k] =  temp;
				}
			}
		}
	}
	
	//Mat ex_roi = curr_frames.frame_view_007(r_007);
	
	//Mat roi_img = curr_frames.frame_view_007.clone();
	//roi_img.setTo(Scalar(0,0,0),Mat());
	
	//curr_frames.frame_view_005.copyTo(roi_img,ex_roi);

	//showImage("Subject Image",ex_roi);
	//showImage("ROI Image 005",this->frame_list.sub_view_005);
	//showImage("ROI Image 007",this->frame_list.sub_view_007);
	//showImage("ROI Image 008",this->frame_list.sub_view_008);
}

IDList MySequence::getIDList(void)
{
	return this->all_ids;
}

void MySequence::clearIDList(void)
{
	this->all_ids.ids_005.clear();
	this->all_ids.ids_007.clear();
	this->all_ids.ids_008.clear();
}

void MySequence::populateIDList(void)
{
	list<Ground_truth> curr_truth = this->getGroundTruth();
	list<Ground_truth>::iterator itr;

	for(itr = curr_truth.begin(); itr != curr_truth.end(); itr++)
	{
		if(itr->isPresent_005 == true)
			this->all_ids.ids_005.push_back(itr->sub_id);
		
		if(itr->isPresent_007 == true)
			this->all_ids.ids_007.push_back(itr->sub_id);

		if(itr->isPresent_008 == true)
			this->all_ids.ids_008.push_back(itr->sub_id);
	}

}
Size MySequence::getImageSize(void)
{
	return this->getFramesParallel().frame_view_005.size();
}
