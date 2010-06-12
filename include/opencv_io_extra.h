#ifndef __ACVHW2_IO_H
#define __ACVHW2_IO_H

#include <cv.h>
#include <string.h>
#include <iostream>

using std::string;

namespace cv
{
    static inline void write( cv::FileStorage& fs, const cv::PCA & pca_obj ){
        string name("pca");
        string mean_name = name + "_mean";
        string eigenvalues_name = name + "_eigenvalues";
        string eigenvectors_name = name + "_eigenvectors";

        write(fs, mean_name, pca_obj.mean);
        write(fs, eigenvalues_name, pca_obj.eigenvalues);
        write(fs, eigenvectors_name, pca_obj.eigenvectors);
    }

    static inline void read( cv::FileStorage& fs, cv::PCA & pca_obj){
        read(fs["pca_mean"], pca_obj.mean, Mat());
        read(fs["pca_eigenvalues"], pca_obj.eigenvalues, Mat());
        read(fs["pca_eigenvectors"], pca_obj.eigenvectors, Mat());

        if (pca_obj.mean.empty() || pca_obj.eigenvalues.empty() || pca_obj.eigenvectors.empty()){
            std::cout << "Data read is empty. Check input filename " << fs.elname << std::endl;
        }
    }
}

#endif
