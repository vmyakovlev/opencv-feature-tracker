#ifndef __BLOB_DETECTOR_
#define __BLOB_DETECTOR_

#include "common.hpp"
#include "Blob.h"
#include <vector>

namespace cv{
    class Mat;

    class BlobDetector{
    public:
        BlobDetector();
        //BlobDetector(); // fully parameter specification
        virtual std::vector<Blob> operator()(const Mat & input_foreground_mask_image) const;
    private:
        DISALLOW_COPY_AND_ASSIGN(BlobDetector);
    };
}

#endif
