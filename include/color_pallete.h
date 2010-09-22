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
#ifndef __COLOR_PALLETE_
#define __COLOR_PALLETE_

#include <cv.h>

namespace ColorPallete{
    static int NUM_COLORS_IN_PALLETE = 20;
    static CvScalar colors [] = {
        //http://www.colourlovers.com/palette/1236980/Eggplant_and_Orange
        CV_RGB(244,238,203),
        CV_RGB(247,198,234),
        CV_RGB(247,189,65),
        CV_RGB(247,70,195),
        CV_RGB(110,59,103),
        //http://www.colourlovers.com/palette/1236977/christina_darling
        CV_RGB(157,33,33),
        CV_RGB(44,24,25),
        CV_RGB(237,195,147),
        CV_RGB(205,141,113),
        CV_RGB(236,217,187),
        //http://www.colourlovers.com/palette/1236972/M.A.T.H
        CV_RGB(160,173,147),
        CV_RGB(5,5,5),
        CV_RGB(34,33,41),
        CV_RGB(238,237,247),
        CV_RGB(78,78,87),
        //http://www.colourlovers.com/palette/1236968/Rotom_Wash
        CV_RGB(136,184,240),
        CV_RGB(72,120,248),
        CV_RGB(8,72,192),
        CV_RGB(72,72,80),
        CV_RGB(232,112,40),
        //http://www.colourlovers.com/palette/1236956/Flower_Field
        CV_RGB(234,230,203),
        CV_RGB(252,77,108),
        CV_RGB(255,184,178),
        CV_RGB(107,150,96),
        CV_RGB(53,109,72)

    };

}

#endif
