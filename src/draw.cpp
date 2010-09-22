/* This is the contributed code:

Original Version: 2010-09-21  Dat Chu dattanchu@gmail.com
Original Comments:

Provides extra methods for drawing the following
- a hull
- a polyline
- a rotated rect

*/

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

#include "draw.h"
#include "misc.h"

/** \brief Draw line connecting the points on a convex hull

  You can use it for drawing lines connecting point pair as well
*/
void draw_hull(Mat im, const vector<Point> hull_points, CvScalar color){
    int npts = hull_points.size();

    for (int i=0; i<npts-1; i++){
        line(im, hull_points[i], hull_points[i+1], color);
    }
    // connect the final pair
    line(im, hull_points[npts-1], hull_points[0], color);
}

void draw_hull(Mat im, const vector<Point2f> hull_points, CvScalar color){
    vector<Point> points(hull_points.size());

    for (size_t i=0; i<hull_points.size(); i++){
        points[i] = hull_points[i];
    }
    draw_hull(im, points, color);
}

/** \brief Draw this rotated rectangle onto our image with this color

  This is just a special case of draw_hull where the number of points on the hull is 4

  \see draw_hull
*/
void rotated_rect(Mat im, const RotatedRect & rot_rect, CvScalar color){
    CvPoint2D32f box_vtx[4];
    cvBoxPoints(rot_rect, box_vtx);

    // copied shamelessly from minarea.c
    // it initialize to the last point, then connect to point 0, point 1, point 2 pair-wise
    CvPoint pt0, pt;
    pt0.x = cvRound(box_vtx[3].x);
    pt0.y = cvRound(box_vtx[3].y);
    for(int i = 0; i < 4; i++ )
    {
        pt.x = cvRound(box_vtx[i].x);
        pt.y = cvRound(box_vtx[i].y);
        line(im, pt0, pt, color, 1, CV_AA, 0);
        pt0 = pt;
    }
}

/** \brief A C++ version of polylines implemented within OpenCV
*/
void polylines(Mat im, vector<vector<Point2f> > points, bool isClosed, const Scalar & color, int thickness /*= 1*/, int lineType/*=8*/, int shift/*=0*/){
    // Because drawing points => expect Point**, we will need to make some conversion
    // TODO: add method in OpenCV that takes vector<Point> instead of just Point*
    Point ** points_to_draw = vec_vec_to_arr_arr<Point>(points);

    int * npts = new int[points.size()];

    // find number of points for each contour
    for (int i = 0; i < points.size() ; i++)
        npts[i] = points[i].size();

    // draw
    polylines(im, const_cast<const Point**>(points_to_draw), npts, points.size(), isClosed, color, thickness, lineType, shift);

    // clean up the memory mess :)
    delete_arr_arr<Point>(points_to_draw, points.size());
    delete [] npts;
}
