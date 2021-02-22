#pragma once
#include "bounding_box.h"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
/**
 * @brief Manage feature points for optical flow tracking
 */
class FeaturePointsManager {
   public:
    FeaturePointsManager() = default;

    void initialize(cv::Mat img, BoundingBox initial_bbox);

    void extract_new_feature_points(cv::Mat img);

    void process_feature_points(cv::Mat img, const std::vector<cv::Point2f>& feature_points,
                                std::vector<uchar>& status);

    std::vector<cv::Point2f> get_feature_points() const { return feature_points_; };

    BoundingBox get_bbox() const { return bbox_; };

   private:
    cv::Mat compute_mask(int rows, int cols);

    void update_status(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status);

    void update_bbox(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status, int width_img, int height_img);

    void update_feature_points(const std::vector<cv::Point2f>& new_feature_points, std::vector<uchar>& status);

    void mark_status_with_amplitude(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status, float rate);

    void mark_status_with_angle(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status, float rate);

    void mark_status_with_contained_points(std::vector<uchar>& status);

    void visualize(cv::Mat img, const std::vector<cv::Point2f>& feature_points_at_new_position);
 
    bool is_enough_points() const { return feature_points_.size() > 25; };

    std::vector<cv::Point2f> feature_points_;
    BoundingBox bbox_;
};

std::vector<cv::Point2f>& operator+=(const std::vector<cv::Point2f>& feature_points_1,
                                     const std::vector<cv::Point2f>& feature_points_2);