#ifndef __SPLITTER_H
#define __SPLITTER_H

#include <cv.h>
#include <vector>

/**
	Indicates how the rectangular region will be splitted:
	NONE: it will not be splitted
	UL: it will be splitted into upper half and lower half
	ULF: it will be splitted into upper half, lower half and "foot"
 */
enum  SplitMode {NONE, UL, ULF};

class Splitter {
public:
	void SplitMat(const cv::Mat& mat_in, SplitMode mode, std::vector<cv::Mat>* mats_out);
};

#endif