#include "svm_classifier.h"
#include <iostream>

using std::cout;
using std::endl;

SVMClassifier::SVMClassifier(int dim)
	: feat_dim_(dim)
{ 
	num_of_models_ = 0; 
	classifier_ready_ = false;
}

SVMClassifier::~SVMClassifier()
{
	for (int i = 0; i < num_of_models_; ++i) {
		
		if (svm_models_[i] != NULL)
			svm_destroy_model(svm_models_[i]);

		if (svm_nodes_[i] != NULL)
			delete [] svm_nodes_[i];
	}
}

int SVMClassifier::Classify( const cv::Mat& descriptor )
{
	assert(descriptor.rows == 1);
	assert(descriptor.cols == feat_dim_);

	if (!classifier_ready_ || num_of_models_ == 0) {
		cout << "Classifier not ready." << endl;
		return 0;
	}
	
	double prob_estimates[2];
	double current_max_prob = 0;
	int id = 0;
	for (int i = 0; i < num_of_models_; ++i) {
		svm_model* svm_model = svm_models_[i];
		svm_node* svm_nodes = svm_nodes_[i];
		for (int j = 0; j < feat_dim_; ++j) {
			svm_nodes[j].index = j;
			svm_nodes[j].value = descriptor.at<float>(0, j);
		}
		svm_nodes[feat_dim_].index = -1;

		double predict_label = svm_predict_probability(svm_model, svm_nodes, prob_estimates);
		if (predict_label == 1 && prob_estimates[0] > current_max_prob) {
			id = i + 1;
			current_max_prob = prob_estimates[0];
		}
	}	

	return id;
}

bool SVMClassifier::LoadModels( const std::vector<std::string>& model_files )
{
	int num_of_files = model_files.size();
	for (int i = 0; i < num_of_files; ++i) {
		svm_model* svm_model = svm_load_model(model_files[i].c_str());
		if (svm_model == NULL) {
			cout << "Could not load SVM model from " << model_files[i] << endl;
			return false;
		}
		svm_node* svm_nodes = new svm_node[feat_dim_+1];

		svm_models_.push_back(svm_model);
		svm_nodes_.push_back(svm_nodes);
	}

	classifier_ready_ = true;
	num_of_models_ = svm_models_.size();

	return true;
}