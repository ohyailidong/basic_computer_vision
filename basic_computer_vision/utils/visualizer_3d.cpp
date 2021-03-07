/**
______________________________________________________________________
*********************************************************************
* @brief  This file is developed for the course of ShenLan XueYuan:
* Fundamental implementations of Computer Vision
* all rights preserved
* @author Xin Jin, Zhaoran Wu
* @contact: xinjin1109@gmail.com, zhaoran.wu1@gmail.com
*
______________________________________________________________________
*********************************************************************
**/
#include "visualizer_3d.h"
#include <opencv2/highgui.hpp>

Vis3D::Vis3D()
    : window_("features in bgr space"),
      x_axis_(cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(255.0, 0.0, 0.0), 0.02, cv::viz::Color::blue()),
      y_axis_(cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(0.0, 255.0, 0.0), 0.02, cv::viz::Color::green()),
      z_axis_(cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(0.0, 0.0, 255.0), 0.02, cv::viz::Color::red()),
      cube_(cv::Point3d(0, 0, 0), cv::Point3d(255.0, 255.0, 255.0)) {
}

void Vis3D::visualize() {
    window_.showWidget("x_axis", x_axis_);
    window_.showWidget("y_axis", y_axis_);
    window_.showWidget("z_axis", z_axis_);
    window_.showWidget("cube", cube_);

    while (!window_.wasStopped()) {
        if (ptr_features_) {
            std::lock_guard<std::mutex> lg_features(mutex_features_);
            if (ptr_features_) {
                window_.showWidget("features", *ptr_features_);
            }
        }
        cv::Mat vis;

        {
            std::lock_guard<std::mutex> lg_origin_img(mutex_img_);
            cv::hconcat(origin_img_, curr_img_, vis);
        }

        cv::imshow("right: original img, left: segmentation live result", vis);
        cv::waitKey(1);

        window_.spinOnce(1, true);
    }
}

void Vis3D::set_features(cv::Mat points_mat) {
    cv::Mat color_mat;
    points_mat.convertTo(color_mat, CV_8UC3);

    if (!points_mat.empty() && !color_mat.empty()) {
        std::lock_guard<std::mutex> lg_features(mutex_features_);
        ptr_features_ = std::make_shared<cv::viz::WCloud>(points_mat, color_mat);
    }

    // make segmentation img
    std::lock_guard<std::mutex> lg_origin_img(mutex_img_);
    curr_img_ = color_mat;
}

void Vis3D::set_origin_img(cv::Mat origin_img) {
    std::lock_guard<std::mutex> lg_origin_img(mutex_img_);
    curr_img_ = origin_img_ = origin_img;
}
