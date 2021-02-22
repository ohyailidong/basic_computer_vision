#pragma once

#include "bounding_box.h"
#include "feature_points_manager.h"
#include <opencv2/core/core.hpp>
/**
 * @brief class for tracking algorithm based on optical flow method
 *
 */
class OpticalFlowTracker {
   public:
    OpticalFlowTracker() = default;

    void process(BoundingBox initial_bbox, const std::vector<cv::Mat>& video);

   private:
    FeaturePointsManager feature_points_manager_;
};