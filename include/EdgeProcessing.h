#pragma once

#include "MySequence.h"
#include <vector>

struct Edges
{
	Mat edges_005, edges_007, edges_008;
};

struct Lines
{
	vector<Vec4i> lines_005, lines_007, lines_008;
	Mat line_map_005, line_map_007, line_map_008;
};

struct subLines
{
	int frame_no, /*view_no,*/ sub_id;
	string view_no;
	//int x1, y1, x2, y2;
	Point p1, p2;
	double theta, slope, length;
	bool isPresent;

	Point p1a, p1b, p2a, p2b;
	double band_dist;
	vector<Point> band_points;
	Rect band_rect;
};

struct subjectLines
{
	list<subLines> sub_line_list;
	int sub_id;
};

struct GroupedLines
{
	list<subjectLines> line_list_005;
	list<subjectLines> line_list_007;
	list<subjectLines> line_list_008;
};

struct Roots
{
	double root1, root2;
	Roots(void);
	Roots(double, double);
};

class EdgeProcessing: public MySequence
{
private:

	//Mat edge_map;//,line_map; 
	Edges all_edges;
	Lines all_lines;
	
	subLines sub_lines;  // dummy
	list<subLines> line_list; 

	subjectLines subject_lines; //dummy
	GroupedLines grouped_lines; //use for subject wise line assignment and subject to subject matching
	 
	static Mat getCanny(Mat,double,double,int,bool); 
	static vector<Vec4i> getHoughLinesP(Mat,double,double,int,double,double);
	static void showLines(vector<Vec4i>, Mat); // (vector of lines, line map) // this should be a public member
	static vector<Vec4i> eliminateLines(vector<Vec4i>);  // (vector of lines for current view)
	void initLineList(void);
	//void populateLineList(int);
	void assignLinesUsingGroundTruth(Point, Point, string); // (x1,y1, x2,y2, view_no)
	static bool doesRectContainPoint(list<subLines>::iterator, Point); // (itr has four corner points, Query point)
	static bool compareListElems(subLines, subLines);
	//void createSubLists(void);
	void calculateLineParams(void); //for calculating slope, length, etc.
	void calculateBandPoints(list<subLines>::iterator);
	void calculateBand(list<subLines>::iterator);
	static Roots solveQuadratic(double, double, double);
	static double calculateTriangleArea(Point, Point, Point); // (three verices of triangle)
	static void renameAdjacentVertices(list<subLines>::iterator);
	static double distanceFormula(Point, Point);


public: //write getter methods
	EdgeProcessing(void);
	EdgeProcessing(string);
	~EdgeProcessing(void);
	Edges getEdges(void);
	Lines getLines(void);
	list<subLines> getLineList(void);
	GroupedLines getGroupedLines(void);
	void showLines(string); // (view_no) //one view at a time please!(don't use "all")
	void clearAllLines(void);
	void edgeDetection(int, string); //(frame_no,view_no)
	void lineDetection(int, string); //(frame_no,view_no)
	void classifyLines(string); //(view_no)
	void populateLineList(int); //(frame_no)
	void initLineMaps(void);
	void createSubLists(void);
	void extractBands(void); // work on complete list (before grouping)
	
};
