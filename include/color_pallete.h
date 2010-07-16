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
