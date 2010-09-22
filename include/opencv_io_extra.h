/*///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//*/
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
