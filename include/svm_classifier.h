#ifndef __SVM_CLASSIFIER_H
#define __SVM_CLASSIFIER_H

#include <string>
#include <vector>
#include <cv.h>
#include "svm.h"

class SVMClassifier {
public:
	SVMClassifier(int dim);
	~SVMClassifier();

	bool LoadModels(const std::vector<std::string>& model_files);
	int Classify(const cv::Mat& descriptor);

private:
	std::vector<svm_model*> svm_models_;
	std::vector<svm_node*> svm_nodes_;
	int feat_dim_;
	int num_of_models_;
	bool classifier_ready_;
};

#endif