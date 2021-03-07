#pragma once
#include <opencv2/core/core.hpp>

class MotionPredictor {
   public:
    MotionPredictor(cv::Point2f initial_pos) : curr_pos_(initial_pos), last_pos_(initial_pos) {
    }

    cv::Point2f get_curr_pos() const {
        return curr_pos_;
    }

    cv::Point2f next_pos() {
        curr_pos_.x += curr_pos_.x - last_pos_.x;
        curr_pos_.y += curr_pos_.y - last_pos_.y;
        return curr_pos_;
    }

    void set_observation(cv::Point2f pos) {
        last_pos_ = curr_pos_;
        curr_pos_ = pos;
    }

   private:
    cv::Point2f last_pos_;
    cv::Point2f curr_pos_;
};