#pragma once
#include "../utils//visualizer_3d.h"
#include <memory>
#include <opencv2/core/core.hpp>

class MeanShiftSeg {
   public:

    MeanShiftSeg(double radius, Vis3D* vis_ptr);
    void process(cv::Mat img);

   private:
    void update_mass_center();
    double radius_square_;  // squre radius
    cv::Mat features_origin_;  // originial input, which will not changed during the segmentation
    cv::Mat features_curr_;    // record the new position of all the features

    std::unique_ptr<Vis3D> visualizer;
};