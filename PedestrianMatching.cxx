
#include <iostream>
#include <string>
#include "cv.h"
#include "highgui.h"
#include "MySequence.h"
#include "EdgeProcessing.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
	//MySequence S; 
	//S.setPath();
	//S.setFramesParallel(300);
	//S.setFramesSingle(300,7);

	EdgeProcessing E;
	int frame_count = 5;
	string view_no = "5"; //be careful!!

	//while(frame_count<70) // clear edges from past frames, not working in loop without debugger
	//{
		frame_count++;	
		E.setFramesParallel(frame_count);
		E.setGroundTruth(frame_count);
		E.useGroundTruth(frame_count);
		E.edgeDetection(frame_count, view_no);
		E.lineDetection(frame_count, view_no);
		E.classifyLines(view_no);
		E.populateLineList(frame_count);
		E.extractBands();

		E.createSubLists(); // call this guy after all the full-list operations are done
		
		E.showLines("5");
		E.showLines("7");
		E.showLines("8");
		
		E.clearAllLines();
		E.clearGroundTruth(frame_count); //call in destructor?
		waitKey(0);
	//}
	
	//To do: Store lines, descriptors per view per subject using ground truth, classify lines and
		//build descriptors
		// Work on StarDetector


	
	waitKey(0);
    return 0;
	//getch();
}
