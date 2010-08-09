#ifndef __CV_FEATURE_TRACKER_TRACKER_HPP
#define __CV_FEATURE_TRACKER_TRACKER_HPP

namespace cv {
    /** \class Tracker

      An interface for a tracker class. A tracker keeps track of different information

      How do we make this class accessible without knowing what type of data is being kept?
      We can use template to decide the datatype. But that would increase compilation time and
      make code harder to follow. What other options do we have?
    */
    class Tracker {

    };
}

#endif
