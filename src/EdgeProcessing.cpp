

#include "EdgeProcessing.h"

using namespace cv;

EdgeProcessing::EdgeProcessing(void): MySequence()
{
	//this->all_edges.edges_005.setTo(Scalar(0),Mat());//this->frame_list.frame_view_005.clone();
	//this->all_edges.edges_007.setTo(Scalar(0),Mat());
	//this->all_edges.edges_008.setTo(Scalar(0),Mat());
	//cout<<this->all_edges.edges_005.channels();

	//this->all_edges.edges_005.create(this->getFramesParallel().frame_view_005.size(),CV_8UC1); 

	//this->all_edges.edges_005.create(this->frame_list.frame_view_005.size(),CV_8UC1);
	this->all_edges.edges_005.create(576,720,CV_8UC1);
	//this->all_edges.edges_007.create(this->frame_list.frame_view_007.size(),CV_8UC1);
	//this->all_edges.edges_008.create(this->frame_list.frame_view_008.size(),CV_8UC1);
	this->all_edges.edges_007.create(576,720,CV_8UC1);
	this->all_edges.edges_008.create(576,720,CV_8UC1);
	
	//this->all_lines.line_map_005.create(this->frame_list.frame_view_005.size(),CV_8UC3);
	this->all_lines.line_map_005.create(576,720,CV_8UC3);
	//this->all_lines.line_map_005.create(this->getFramesParallel().frame_view_005.size(),CV_8UC3); 
	this->all_lines.line_map_005.setTo(Scalar(0), Mat());

	//this->all_lines.line_map_007.create(this->frame_list.frame_view_007.size(),CV_8UC3);
	this->all_lines.line_map_007.create(576,720,CV_8UC3);
	this->all_lines.line_map_007.setTo(Scalar(0), Mat());

	//this->all_lines.line_map_008.create(this->frame_list.frame_view_008.size(),CV_8UC3);
	this->all_lines.line_map_008.create(576,720,CV_8UC3);
	this->all_lines.line_map_008.setTo(Scalar(0), Mat());

	this->initLineList();

}

EdgeProcessing::EdgeProcessing(string path ): MySequence(path)
{
	//this->all_edges.edges_005.setTo(Scalar(0),Mat());//this->frame_list.frame_view_005.clone();
	//this->all_edges.edges_007.setTo(Scalar(0),Mat());
	//this->all_edges.edges_008.setTo(Scalar(0),Mat());
	//cout<<this->all_edges.edges_005.channels();

	//this->all_edges.edges_005.create(this->getFramesParallel().frame_view_005.size(),CV_8UC1); 

	//this->all_edges.edges_005.create(this->frame_list.frame_view_005.size(),CV_8UC1);
	this->all_edges.edges_005.create(576,720,CV_8UC1);
	//this->all_edges.edges_007.create(this->frame_list.frame_view_007.size(),CV_8UC1);
	//this->all_edges.edges_008.create(this->frame_list.frame_view_008.size(),CV_8UC1);
	this->all_edges.edges_007.create(576,720,CV_8UC1);
	this->all_edges.edges_008.create(576,720,CV_8UC1);
	
	//this->all_lines.line_map_005.create(this->frame_list.frame_view_005.size(),CV_8UC3);
	this->all_lines.line_map_005.create(576,720,CV_8UC3);
	//this->all_lines.line_map_005.create(this->getFramesParallel().frame_view_005.size(),CV_8UC3); 
	this->all_lines.line_map_005.setTo(Scalar(0), Mat());

	//this->all_lines.line_map_007.create(this->frame_list.frame_view_007.size(),CV_8UC3);
	this->all_lines.line_map_007.create(576,720,CV_8UC3);
	this->all_lines.line_map_007.setTo(Scalar(0), Mat());

	//this->all_lines.line_map_008.create(this->frame_list.frame_view_008.size(),CV_8UC3);
	this->all_lines.line_map_008.create(576,720,CV_8UC3);
	this->all_lines.line_map_008.setTo(Scalar(0), Mat());

	this->initLineList();
}

EdgeProcessing::~EdgeProcessing(void)
{
}

Edges EdgeProcessing::getEdges(void)
{
	return this->all_edges;
}
Lines EdgeProcessing::getLines(void)
{
	return this->all_lines;
}
void EdgeProcessing::edgeDetection(int count, string view_no)
{
	Mat gray_frame, curr_frame, curr_frame_005, curr_frame_007, curr_frame_008;
	
	double t1 = 100; //recommended t1:t2::2:1,3:1,5:1,3:2
	double t2 = 50;
	int size = 3;
	bool L2Gradient = true;
	
	Frames all_frames = this->getFramesParallel();   
	
	//this->all_edges.edges_007.create(576,720,CV_8UC1);

	if(view_no == "5")
	{
		if(this->getGroundTruthFlag() == true)
			curr_frame = all_frames.sub_view_005.clone(); 
		else
			curr_frame = all_frames.frame_view_005.clone();
	
		this->all_edges.edges_005 = getCanny(curr_frame,t1,t2,size,L2Gradient);
		//showImage("Edges view_005",this->all_edges.edges_005);
	}
	else if(view_no == "7")
	{
		if(this->getGroundTruthFlag() == true)
			curr_frame = all_frames.sub_view_007.clone();  
		else
			curr_frame = all_frames.frame_view_007.clone();
	
		this->all_edges.edges_007 = getCanny(curr_frame,t1,t2,size,L2Gradient);
		//showImage("Edges view_007",this->all_edges.edges_007);
	}
	else if(view_no == "8")
	{
		if(this->getGroundTruthFlag() == true)
			curr_frame = all_frames.sub_view_008.clone();  
		else
			curr_frame = all_frames.frame_view_008.clone();
	
		this->all_edges.edges_008 = getCanny(curr_frame,t1,t2,size,L2Gradient);
		//showImage("Edges view_008",this->all_edges.edges_008);
	}
	else if(view_no == "all")
	{
		if(this->getGroundTruthFlag() == true)
		{
			curr_frame_005 = all_frames.sub_view_005.clone();
			curr_frame_007 = all_frames.sub_view_007.clone();
			curr_frame_008 = all_frames.sub_view_008.clone();
		}
		else
		{
			curr_frame_005 = all_frames.frame_view_005.clone();
			curr_frame_007 = all_frames.frame_view_007.clone();
			curr_frame_008 = all_frames.frame_view_008.clone();
		}

		this->all_edges.edges_005 = getCanny(curr_frame_005,t1,t2,size,L2Gradient);
		this->all_edges.edges_007 = getCanny(curr_frame_007,t1,t2,size,L2Gradient);
		this->all_edges.edges_008 = getCanny(curr_frame_008,t1,t2,size,L2Gradient);

		//showImage("Edges view_005",this->all_edges.edges_005);
		//showImage("Edges view_007",this->all_edges.edges_007);
		//showImage("Edges view_008",this->all_edges.edges_008);
	}
	else
	{
		cout<<"Incorrect view number!"<<endl;
		this->all_edges.edges_005.setTo(Scalar(0), Mat());
		this->all_edges.edges_007.setTo(Scalar(0), Mat());
		this->all_edges.edges_008.setTo(Scalar(0), Mat());
	}

}

Mat EdgeProcessing::getCanny(Mat curr_frame,double t1, double t2, int size, bool L2Gradient)
{
	Mat gray_frame;
	Mat edge_map;
	
	//double t1 = 100; //recommended t1:t2::2:1,3:1,5:1,3:2
	//double t2 = 50;
	//int size = 3;
	//bool L2Gradient = true;

	gray_frame.create(curr_frame.size(),curr_frame.type());

	cvtColor(curr_frame,gray_frame,CV_RGB2GRAY,0);

	edge_map.create(gray_frame.size(),gray_frame.type());
	edge_map.setTo(Scalar(0),Mat());

	Canny(gray_frame,edge_map,t1,t2,size,L2Gradient);

	return edge_map;
}


void EdgeProcessing::lineDetection(int count, string view_no)
{
	//vector<Vec4i> lines;
	//cout<<lines.max_size()<<endl;
	Edges curr_edges = this->getEdges();
	//vector<Vec4i> lines = this->getLines().lines_005;

	double rho_res = 1; //in pixels
	double theta_res = CV_PI/180; //in radians
	int acc_thresh = 5;
	double min_length = 10; 
	double max_gap = 5; //try 15 to get more lines

	if(view_no == "5")
	{
		this->all_lines.lines_005 = getHoughLinesP(curr_edges.edges_005.clone(),rho_res,theta_res,acc_thresh,min_length,max_gap);	
		showLines(this->all_lines.lines_005,this->all_lines.line_map_005);
		//showImage("Lines view_005",this->all_lines.line_map_005);
		//showLines(this->all_lines.lines_005,this->getFramesParallel().frame_view_005);
		//showImage("Lines view_005",this->getFramesParallel().frame_view_005);
	}
	else if(view_no == "7")
	{
		this->all_lines.lines_007 = getHoughLinesP(curr_edges.edges_007.clone(),rho_res,theta_res,acc_thresh,min_length,max_gap);	
		showLines(this->all_lines.lines_007,this->all_lines.line_map_007);
		//showImage("Lines view_007",this->all_lines.line_map_007);
		//showLines(this->all_lines.lines_007,this->getFramesParallel().frame_view_007);
		//showImage("Lines view_007",this->getFramesParallel().frame_view_007);

	}
	else if(view_no == "8")
	{
		this->all_lines.lines_008 = getHoughLinesP(curr_edges.edges_008.clone(),rho_res,theta_res,acc_thresh,min_length,max_gap);	
		showLines(this->all_lines.lines_008,this->all_lines.line_map_008);
		//showImage("Lines view_008",this->all_lines.line_map_008);
		//showLines(this->all_lines.lines_008,this->getFramesParallel().frame_view_008);
		//showImage("Lines view_008",this->getFramesParallel().frame_view_008);
	}
	else if(view_no == "all")
	{
		this->all_lines.lines_005 = getHoughLinesP(curr_edges.edges_005.clone(),rho_res,theta_res,acc_thresh,min_length,max_gap);	
		showLines(this->all_lines.lines_005,this->all_lines.line_map_005);
		//showImage("Lines view_005",this->all_lines.line_map_005);
		//showLines(this->all_lines.lines_005,this->getFramesParallel().frame_view_005);
		//showImage("Lines view_005",this->getFramesParallel().frame_view_005);

		this->all_lines.lines_007 = getHoughLinesP(curr_edges.edges_007.clone(),rho_res,theta_res,acc_thresh,min_length,max_gap);	
		showLines(this->all_lines.lines_007,this->all_lines.line_map_007);
		//showImage("Lines view_007",this->all_lines.line_map_007);
		//showLines(this->all_lines.lines_007,this->getFramesParallel().frame_view_007);
		//showImage("Lines view_007",this->getFramesParallel().frame_view_007);

		this->all_lines.lines_008 = getHoughLinesP(curr_edges.edges_008.clone(),rho_res,theta_res,acc_thresh,min_length,max_gap);	
		showLines(this->all_lines.lines_008,this->all_lines.line_map_008);
		//showImage("Lines view_008",this->all_lines.line_map_008);
		//showLines(this->all_lines.lines_008,this->getFramesParallel().frame_view_008);
		//showImage("Lines view_008",this->getFramesParallel().frame_view_008);
	}
	else 
	{
		cout<<"Incorrect view_no"<<endl;

		showImage("Lines view_005",this->all_lines.line_map_005);
		showImage("Lines view_007",this->all_lines.line_map_007);
		showImage("Lines view_008",this->all_lines.line_map_008);
	}	
}
vector<Vec4i> EdgeProcessing::getHoughLinesP(Mat frame, double rho, double theta, int acc, double min_length, double max_gap)
{
	vector<Vec4i> lines;

	HoughLinesP(frame,lines,rho,theta,acc,min_length,max_gap);
	return lines;
}

void EdgeProcessing::showLines(vector<Vec4i> lines, Mat line_map)
{
    for(size_t i = 0; i<lines.size(); i++)
		line(line_map,Point(lines[i][0],lines[i][1]),Point(lines[i][2],lines[i][3]),Scalar(0,0,255),1,8,0);
}
void EdgeProcessing::classifyLines(string view_no)
{
	if(view_no == "5")
	{
		this->all_lines.lines_005 = eliminateLines(this->all_lines.lines_005); 
		//showLines(this->getLines().lines_005,this->getFramesParallel().frame_view_005);
		//showImage("Classified Lines view_005",this->getFramesParallel().frame_view_005);
	}
	else if(view_no == "7")
	{
		this->all_lines.lines_007 = eliminateLines(this->all_lines.lines_007); 
		//showLines(this->getLines().lines_007,this->getFramesParallel().frame_view_007);
		//showImage("Classified Lines view_007",this->getFramesParallel().frame_view_007);
	}
	else if(view_no == "8")
	{
		this->all_lines.lines_008 = eliminateLines(this->all_lines.lines_008); 
		//showLines(this->getLines().lines_008,this->getFramesParallel().frame_view_008);
		//showImage("Classified Lines view_008",this->getFramesParallel().frame_view_008);
	}
	else if(view_no == "all")
	{
		this->all_lines.lines_005 = eliminateLines(this->all_lines.lines_005); 
		//showLines(this->getLines().lines_005,this->getFramesParallel().frame_view_005);
		//showImage("Classified Lines view_005",this->getFramesParallel().frame_view_005);

		this->all_lines.lines_007 = eliminateLines(this->all_lines.lines_007); 
		//showLines(this->getLines().lines_007,this->getFramesParallel().frame_view_007);
		//showImage("Classified Lines view_007",this->getFramesParallel().frame_view_007);

		this->all_lines.lines_008 = eliminateLines(this->all_lines.lines_008); 
		//showLines(this->getLines().lines_008,this->getFramesParallel().frame_view_008);
		//showImage("Classified Lines view_008",this->getFramesParallel().frame_view_008);
	}
	else
	{
		cout<<"Incorrect view number!"<<endl;
	}

	
}

vector<Vec4i> EdgeProcessing::eliminateLines(vector<Vec4i> lines)
{
	double x_diff = 200.0;
	int max_length = 50;
	double min_angle = 70.0; //degree
	//this is filtering loop, call methods here
    for(size_t i = 0; i<lines.size(); i++)
	{
		
		if(lines[i][0] == lines[i][2])
			lines.erase(lines.begin()+i);
		
		//if(abs(lines[i][0]-lines[i][2]) < x_diff)
		//	lines.erase(lines.begin()+i);
		
		/*double d = sqrt(pow((lines[i][0] - lines[i][2]),2.0) + pow((lines[i][1] - lines[i][3]),2.0));
		if(d > max_length)
			lines.erase(lines.begin()+i);		
		
		double m = (lines[i][1] - lines[i][3])/(lines[i][0] - lines[i][2] + 0.00001);
		double ang = atan(m);	
		double ang_deg = ang*(180/CV_PI);
		//cout<<"line"<<i<<":"<<ang_deg<<endl;

		if(abs(ang_deg) < min_angle)
		{
			//cout<<"line"<<i<<":"<<ang_deg<<endl;
			lines.erase(lines.begin()+i);		
		}*/

	}

	return lines;

}
void EdgeProcessing::initLineList(void)
{
	/*this->sub_lines.x1 = 0;
	this->sub_lines.x2 = 0;
	this->sub_lines.y1 = 0;
	this->sub_lines.y2 = 0;*/

	this->sub_lines.p1 = Point(0,0);
	this->sub_lines.p2 = Point(0,0);
	
	this->sub_lines.p1a = Point(0,0);
	this->sub_lines.p2a = Point(0,0);
	this->sub_lines.p1b = Point(0,0);
	this->sub_lines.p1b = Point(0,0);

	this->sub_lines.frame_no = 0;
	this->sub_lines.view_no = " ";
	this->sub_lines.sub_id = 0;
	this->sub_lines.length = 0.0;
	this->sub_lines.slope = 0.0;
	this->sub_lines.theta = 0.0;
	this->sub_lines.isPresent = true;

	this->sub_lines.band_dist = 0.0; 
	this->sub_lines.band_rect = Rect(0,0,0,0);
}
list<subLines> EdgeProcessing::getLineList(void)
{
	return this->line_list;
}

void EdgeProcessing::populateLineList(int count)
{
	vector<Vec4i> lines_005 = this->all_lines.lines_005;
	vector<Vec4i> lines_007 = this->all_lines.lines_007;
	vector<Vec4i> lines_008 = this->all_lines.lines_008;
	
    for(size_t i=0; i<lines_005.size(); i++)
	{
		this->assignLinesUsingGroundTruth(Point(lines_005[i][0],lines_005[i][1]),Point(lines_005[i][2],lines_005[i][3]), "5");
	}

    for(size_t i=0; i<lines_007.size(); i++)
	{
		this->assignLinesUsingGroundTruth(Point(lines_007[i][0],lines_007[i][1]),Point(lines_007[i][2],lines_007[i][3]), "7");
	}

    for(size_t i=0; i<lines_008.size(); i++)
	{
		this->assignLinesUsingGroundTruth(Point(lines_008[i][0],lines_008[i][1]),Point(lines_008[i][2],lines_008[i][3]), "8");
	}

	this->calculateLineParams(); 
	//showImage("lines005",this->getLines().line_map_005);

	this->line_list.sort(compareListElems);
	//this->createSubLists();
}

void EdgeProcessing::assignLinesUsingGroundTruth(Point p1, Point p2, string view_no)
{
	list<Ground_truth> curr_truth = this->getGroundTruth();
	list<Ground_truth>::iterator itr;

	for(itr = curr_truth.begin(); itr != curr_truth.end(); itr++)
	{
		if(view_no == "5")
		{
			if(itr->isPresent_005 == true)
			{
				if(itr->view_005_rect.contains(p1) == true && itr->view_005_rect.contains(p2) == true)
				{
					this->initLineList();

					this->sub_lines.isPresent = true;
					this->sub_lines.frame_no = itr->frame_no;
					this->sub_lines.p1 = p1;
					this->sub_lines.p2 = p2;
					this->sub_lines.view_no = view_no;
					this->sub_lines.sub_id = itr->sub_id;
														
					this->line_list.push_back(this->sub_lines);
				}
			}
		}
		else if(view_no == "7")
		{
			if(itr->isPresent_007 == true)
			{
				if(itr->view_007_rect.contains(p1) == true && itr->view_007_rect.contains(p2) == true)
				{
					this->initLineList();

					this->sub_lines.isPresent = true;
					this->sub_lines.frame_no = itr->frame_no;
					this->sub_lines.p1 = p1;
					this->sub_lines.p2 = p2;
					this->sub_lines.view_no = view_no;
					this->sub_lines.sub_id = itr->sub_id;
									
					this->line_list.push_back(this->sub_lines);
				}
			}
		}
		else if(view_no == "8")
		{
			if(itr->isPresent_008 == true)
			{
				if(itr->view_008_rect.contains(p1) == true && itr->view_008_rect.contains(p2) == true)
				{
					this->initLineList();

					this->sub_lines.isPresent = true;
					this->sub_lines.frame_no = itr->frame_no;
					this->sub_lines.p1 = p1;
					this->sub_lines.p2 = p2;
					this->sub_lines.view_no = view_no;
					this->sub_lines.sub_id = itr->sub_id;
									
					this->line_list.push_back(this->sub_lines);
				}
			}		
		}
	}
//	this->line_list.sort(compareListElems);
}

GroupedLines EdgeProcessing::getGroupedLines(void)
{
	return this->grouped_lines;
}

bool EdgeProcessing::compareListElems(subLines elem1, subLines elem2)
{
	if(elem1.sub_id < elem2.sub_id)
		return true;
	else
		return false;
}
void EdgeProcessing::createSubLists(void)
{
	int max_subs = 20; //set this global, is this correct
	
	vector<int> line_count(max_subs);

	list<subLines> curr_lines = this->line_list;
	list<subLines> temp_lines_005, temp_lines_007, temp_lines_008;
	list<subLines>::iterator itr;

	//for(int i=1; i<=max_subs; i++)
    for(size_t i = 0; i<line_count.size(); i++)
	{
		for(itr = curr_lines.begin(); itr != curr_lines.end(); itr++)
		{
			if(itr->sub_id == i)
			{
				line_count[i]++;
			}
		}
	}
	
	int local_count = 0;
    for(size_t i = 0; i<line_count.size(); i++)
	{
		if(line_count[i] != 0)
		{
			for(itr = curr_lines.begin(); itr != curr_lines.end(); itr++)
			{
				if(itr->sub_id == i)
				{
					if(itr->view_no == "5")
						temp_lines_005.push_back(*itr);
					if(itr->view_no == "7")
						temp_lines_007.push_back(*itr);
					if(itr->view_no == "8")
						temp_lines_008.push_back(*itr);

					local_count++;
				}
				if(local_count == line_count[i])
				{
					//assign temp_lines to subject_lines
					if(temp_lines_005.empty() == false)
					{
						this->subject_lines.sub_line_list.clear();

						this->subject_lines.sub_id = itr->sub_id;
						this->subject_lines.sub_line_list = temp_lines_005;

						this->grouped_lines.line_list_005.push_back(this->subject_lines);

						temp_lines_005.clear();
					}
					if(temp_lines_007.empty() == false)
					{
						this->subject_lines.sub_line_list.clear();

						this->subject_lines.sub_id = itr->sub_id;
						this->subject_lines.sub_line_list = temp_lines_007;

						this->grouped_lines.line_list_007.push_back(this->subject_lines);

						temp_lines_007.clear();
					}
					if(temp_lines_008.empty() == false)
					{
						this->subject_lines.sub_line_list.clear();

						this->subject_lines.sub_id = itr->sub_id;
						this->subject_lines.sub_line_list = temp_lines_008;

						this->grouped_lines.line_list_008.push_back(this->subject_lines);

						temp_lines_008.clear();
					}
					local_count = 0;
					break;
				}

			}
		}
	}

}
void EdgeProcessing::calculateLineParams(void)
{
	list<subLines> curr_lines = this->line_list;
	list<subLines>::iterator itr;

	double dist = 0.0, theta = 0.0, slope = 0.0, eps = 0.000001;

	for(itr = curr_lines.begin(); itr != curr_lines.end(); itr++)
	{
		itr->length = sqrt(pow((itr->p1.x - itr->p2.x),2.0) + pow((itr->p1.y - itr->p2.y),2.0));
		itr->slope = (itr->p1.y - itr->p2.y)/(itr->p1.x - itr->p2.x + eps);
		itr->theta = atan(itr->slope)*(180/CV_PI);
	}

	this->line_list = curr_lines;
}

void EdgeProcessing::showLines(string view_no)
{
	GroupedLines grp_lines = this->getGroupedLines();
	list<subjectLines> curr_lines;
	list<subjectLines>::iterator itr;
	list<subLines> subj_lines;
	list<subLines>::iterator itr_sub;
	Mat local_map_005,local_map_007,local_map_008;
	//local_map_005 = this->getLines().line_map_005;
	
	this->initLineMaps();

	if(view_no == "5")
	{
		local_map_005 = this->getFramesParallel().frame_view_005.clone();
		//local_map_005 = this->getLines().line_map_005.clone();
		curr_lines = grp_lines.line_list_005;
	}
	if(view_no == "7")
	{
		local_map_007 = this->getFramesParallel().frame_view_007.clone();
		//local_map_007 = this->getLines().line_map_007.clone();
		curr_lines = grp_lines.line_list_007;
	}
	if(view_no == "8")
	{
		local_map_008 = this->getFramesParallel().frame_view_008.clone();
		//local_map_008 = this->getLines().line_map_008.clone();
		curr_lines = grp_lines.line_list_008;
	}

	for(itr = curr_lines.begin(); itr != curr_lines.end(); itr++)
	{
		subj_lines = itr->sub_line_list;
		for(itr_sub = subj_lines.begin(); itr_sub != subj_lines.end(); itr_sub++)
		{
			if(view_no == "5")
			{
				//line(this->getLines().line_map_005,itr_sub->p1,itr_sub->p2,Scalar(0,255,0),1,8,0);
				line(local_map_005,itr_sub->p1,itr_sub->p2,Scalar(0,255,0),1,8,0);
				showImage("lines005",local_map_005);
			}
			if(view_no == "7")
			{
				//line(this->getLines().line_map_007,itr_sub->p1,itr_sub->p2,Scalar(0,255,0),1,8,0);
				line(local_map_007,itr_sub->p1,itr_sub->p2,Scalar(0,255,0),1,8,0);
				showImage("lines007",local_map_007);
			}
			if(view_no == "8")
			{
				//line(this->getLines().line_map_008,itr_sub->p1,itr_sub->p2,Scalar(0,255,0),1,8,0);
				line(local_map_008,itr_sub->p1,itr_sub->p2,Scalar(0,255,0),1,8,0);
				showImage("lines008",local_map_008);
			}
		}
		
	}

	//this->getLines().line_map_005 = local_map_005;
	//showImage("lines005",this->getLines().line_map_005);
	//showImage("lines005",local_map_005);

}

void EdgeProcessing::initLineMaps(void)
{
	this->getLines().line_map_005.setTo(Scalar::all(0), Mat());
	this->getLines().line_map_007.setTo(Scalar::all(0), Mat());
	this->getLines().line_map_008.setTo(Scalar::all(0), Mat());
}
void EdgeProcessing::clearAllLines(void)
{
	this->getLines().lines_005.clear();
	this->getLines().lines_007.clear();
	this->getLines().lines_008.clear();
}

void EdgeProcessing::extractBands(void)
{
	Mat mat_005 = this->getFramesParallel().frame_view_005;
	list<subLines> curr_lines = this->getLineList();
	list<subLines>::iterator itr;

	//int band_dist = 5;

	for(itr = curr_lines.begin(); itr != curr_lines.end(); itr++)
	{
		itr->band_dist = 5.0; 
		
		this->calculateBandPoints(itr);
		this->calculateBand(itr);
		
		//itr->band_rect = boundingRect(Mat(itr->band_points, false));
		//rectangle(mat_005,itr->band_rect.tl(),itr->band_rect.br(),Scalar(255,0,0),1,8,0);
	}

	for(itr = curr_lines.begin(); itr != curr_lines.end(); itr++)
	{
		itr->band_rect = boundingRect(Mat(itr->band_points, false));
		rectangle(mat_005,itr->band_rect.tl(),itr->band_rect.br(),Scalar(255,0,0),1,8,0);
	}
}
void EdgeProcessing::calculateBandPoints(list<subLines>::iterator itr)
{
	double con1 = itr->p1.y + itr->p1.x / itr->slope;
	double con2 = itr->p2.y + itr->p2.x / itr->slope;

	double A1 = 1.0 + 1.0 / pow(itr->slope, 2.0);
	double B1 = (2 * itr->p1.y / itr->slope) - (2 * itr->p1.x) - (2 * con1 / itr->slope);
	double C1 = pow(itr->p1.x, 2.0) + pow(con1, 2.0) - (2 * itr->p1.y * con1) + pow(itr->p1.y, 2.0) - pow(itr->band_dist, 2.0);

	double A2 = A1;
	double B2 = (2 * itr->p2.y / itr->slope) - (2 * itr->p2.x) - (2 * con2 / itr->slope);
	double C2 = pow(itr->p2.x, 2.0) + pow(con2, 2.0) - (2 * itr->p2.y * con2) + pow(itr->p2.y, 2.0) - pow(itr->band_dist, 2.0);
	
	// now I have A1x^2 + B1x + C1 and A2x^2 + B2x + C2 tp solve for two perpendicular lines at endpoints

	Roots roots_p1 = solveQuadratic(A1, B1, C1);
	Roots roots_p2 = solveQuadratic(A2, B2, C2);

	double y1, y2, y3, y4;

	y1 = (-1 * roots_p1.root1 / itr->slope) + con1;
	y2 = (-1 * roots_p1.root2 / itr->slope) + con1;

	y3 = (-1 * roots_p2.root1 / itr->slope) + con2;
	y4 = (-1 * roots_p2.root2 / itr->slope) + con2;

	itr->p1a = Point(ceil(roots_p1.root1), ceil(y1));
	itr->p1b = Point(ceil(roots_p1.root2), ceil(y2));

	itr->p2a = Point(ceil(roots_p2.root1), ceil(y3));
	itr->p2b = Point(ceil(roots_p2.root2), ceil(y4));

	renameAdjacentVertices(itr);
}
Roots::Roots(void)
{
	this->root1 = 0.0;
	this->root2 = 0.0;
}
Roots::Roots(double root1, double root2)
{
	this->root1 = root1;
	this->root2 = root2;
}


Roots EdgeProcessing::solveQuadratic(double A, double B, double C)
{
	double r1, r2;
	double del = pow(B, 2.0) - (4 * A * C);
	
	if(del < 0)
	{
		cout<< "Imaginary Roots!!"<< endl;
		return Roots();
	}

	r1 = ((-1 * B) + sqrt(del)) / (2 * A);
	r2 = ((-1 * B) - sqrt(del)) / (2 * A);
	
	//cout<<r1<<", "<<r2<<endl;

	return Roots(r1, r2);
}


void EdgeProcessing::renameAdjacentVertices(list<subLines>::iterator itr)
{
	double dist1 = distanceFormula(itr->p1a,itr->p2a);
	double dist2 = distanceFormula(itr->p1a,itr->p2b);

	Point temp;
	if(dist1 > dist2)
	{
		swap<Point>(itr->p2a, itr->p2b);
		//temp = itr->p2a;
		//itr->p2a = itr->p2b;
		//itr->p2b = temp;
	}

}

double EdgeProcessing::distanceFormula(Point p1, Point p2)
{
	return sqrt(pow((p1.x - p2.x), 2.0) + pow((p1.y - p2.y), 2.0));
}

void EdgeProcessing::calculateBand(list<subLines>::iterator itr)
{
	Mat mat_005 = this->getFramesParallel().frame_view_005;
	
	//typedef Vec<uchar, 3> data_type;
	//MatConstIterator_<data_type> itr_mat;

	// Using STL-style iterator is not efficient for large data, use simple for loops
	//Point p_ex;
	/*for(itr_mat = mat_005.begin<data_type>(); itr_mat != mat_005.end<data_type>(); ++itr_mat)
	{
		//p_ex = itr_mat.pos();
		//if(this->doesRectContainPoint(itr, itr_mat.pos()) == true)
		//{
			//....
		//}
	}*/

	for(int x = 0; x < mat_005.cols; x++)
	{
		for(int y = 0; y < mat_005.rows; y++)
		{
			if(doesRectContainPoint(itr, Point(x, y)) == true)
			{
				itr->band_points.push_back(Point(x, y));
				//itr->band_rect = boundingRect(Mat(itr->band_points, false));
				//rectangle(mat_005,itr->band_rect.tl(),itr->band_rect.br(),Scalar(255,0,0),1,8,0);
			}
		}
	}

}

bool EdgeProcessing::doesRectContainPoint(list<subLines>::iterator itr, Point p_ex)
{
	double area1 = calculateTriangleArea(itr->p1a, itr->p1b, p_ex);
	double area2 = calculateTriangleArea(itr->p1b, itr->p2b, p_ex);
	double area3 = calculateTriangleArea(itr->p2b, itr->p2a, p_ex);
	double area4 = calculateTriangleArea(itr->p2a, itr->p1a, p_ex);

	if(area1 > 0 && area2 > 0 && area3 > 0 && area4 > 0) 
		return true;
	else if(area1 < 0 && area2 < 0 && area3 < 0 && area4 < 0) 
		return true;
	else 
		return false;
}

double EdgeProcessing::calculateTriangleArea(Point p0, Point p1, Point p2)
{
		//1 | x0 y0 1 |
	//A = - | x1 y1 1 |
		//2 | x2 y2 1 |
	//A = (.5)(x1*y2 - y1*x2 -x0*y2 + y0*x2 + x0*y1 - y0*x1)

	return 0.5 * (p1.x*p2.y - p1.y*p2.x - p0.x*p2.y + p0.y*p2.x + p0.x*p1.y - p0.y*p1.x);
}
