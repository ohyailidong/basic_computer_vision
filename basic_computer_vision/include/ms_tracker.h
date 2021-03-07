#include "bounding_box.h"
#include "motion_predictor.h"
#include "ms_tracking.h"

class MeanShiftTracker {
   public:

    MeanShiftTracker(cv::Mat temp, BoundingBox init_bbox) : motion_predictor(init_bbox.center()), ms_tracking(temp) {
    }

    void process(const std::vector<cv::Mat>& video);

   private:
    MotionPredictor motion_predictor;  // predict the init position according to constant motion model
    MeanShiftTracking ms_tracking;     // tracking the obeject with mean shift
};