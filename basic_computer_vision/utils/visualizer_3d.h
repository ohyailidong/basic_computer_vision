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
#pragma once
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>

class Vis3D {
   public:
    Vis3D();
    /**
     * @brief Set the features object
     *
     * @param [in] points_mat
     */
    void set_features(cv::Mat points_mat);
    /**
     * @brief Set the origin img object
     *
     * @param [in] origin_img
     */
    void set_origin_img(cv::Mat origin_img);
    void visualize();

   private:
    cv::viz::Viz3d window_;

    cv::viz::WArrow x_axis_;
    cv::viz::WArrow y_axis_;
    cv::viz::WArrow z_axis_;

    cv::viz::WCube cube_;

    std::mutex mutex_features_;
    std::shared_ptr<cv::viz::WCloud> ptr_features_ = nullptr;

    std::mutex mutex_img_;
    cv::Mat origin_img_;
    cv::Mat curr_img_;
};