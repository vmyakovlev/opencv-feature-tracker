#include "opencv_io_extra.h"
#include "test_configuration.h"
#include <gtest/gtest.h>

#include <cv.h>
#include <string>
using std::string;
using namespace cv;

TEST(TestIo, PCAWriteRead){
    Mat crazy_mat = Mat::ones(4,5, CV_32F);
    PCA pca(crazy_mat, Mat(), CV_PCA_DATA_AS_ROW);

    FileStorage fs("PCATests.yml", FileStorage::WRITE);
    write(fs, pca);

    fs.release();

    FileStorage read_fs("PCATests.yml", FileStorage::READ);
    PCA new_pca;
    read(fs,pca);

    read_fs.release();

    // Check that we read everything back ok
    ASSERT_EQ(pca.mean.rows, new_pca.mean.rows);
    ASSERT_EQ(pca.mean.cols, new_pca.mean.cols);
    ASSERT_EQ(pca.eigenvalues.rows, new_pca.eigenvalues.rows);
    ASSERT_EQ(pca.eigenvalues.cols, new_pca.eigenvalues.cols);
    ASSERT_EQ(pca.eigenvectors.rows, new_pca.eigenvectors.rows);
    ASSERT_EQ(pca.eigenvectors.cols, new_pca.eigenvectors.cols);
}

TEST(TestIo, PCAReadIntoVector){
    vector<PCA> pcas;
    pcas.resize(2);

    FileStorage fs1(data_folder_path + "/pca1.yml", FileStorage::READ);
    read(fs1, pcas[0]);

    FileStorage fs2(data_folder_path + "/pca1.yml", FileStorage::READ);
    read(fs2,pcas[1]);

    // Check that we read everything back ok
    ASSERT_EQ(pcas[0].mean.rows, 1);
    ASSERT_EQ(pcas[0].mean.cols, 200);
    ASSERT_EQ(pcas[0].eigenvalues.rows, 100);
    ASSERT_EQ(pcas[0].eigenvalues.cols, 1);
    ASSERT_EQ(pcas[0].eigenvectors.rows, 100);
    ASSERT_EQ(pcas[0].eigenvectors.cols, 200);

    ASSERT_EQ(pcas[1].mean.rows, 1);
    ASSERT_EQ(pcas[1].mean.cols, 200);
    ASSERT_EQ(pcas[1].eigenvalues.rows, 100);
    ASSERT_EQ(pcas[1].eigenvalues.cols, 1);
    ASSERT_EQ(pcas[1].eigenvectors.rows, 100);
    ASSERT_EQ(pcas[1].eigenvectors.cols, 200);
}
