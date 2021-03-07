#include "..\include\feature_points_manager.h"
#include "..\utils\math_utils.h"
#include "..\utils\opencv_utils.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <opencv2/video/tracking.hpp>

std::vector<cv::Point2f> extract_feature_points(cv::Mat img, cv::Mat mask, float weight);

std::vector<cv::Vec2f> compute_pixel_motion(const std::vector<cv::Point2f>& old_feature_points,
                                            const std::vector<cv::Point2f>& new_feature_points);

std::vector<cv::Point2f>& operator+=(std::vector<cv::Point2f>& feature_points_1,
                                     std::vector<cv::Point2f>& feature_points_2) {
    for (cv::Point2f point : feature_points_2) {
        feature_points_1.push_back(point);
    }
    return feature_points_1;
}

void FeaturePointsManager::initialize(cv::Mat img, BoundingBox initial_bbox) {
    bbox_ = initial_bbox;
    extract_new_feature_points(img);
}

std::vector<cv::Point2f> extract_feature_points(cv::Mat img, cv::Mat mask, float weight) {
    std::vector<cv::Point2f> feature_points;

    double quality_level = std::max(0.02, weight * 0.5);
    double min_distance = std::max(2.0f, weight * 8);
	int maxCorners = 200;
    cv::goodFeaturesToTrack(img, feature_points, maxCorners, quality_level, min_distance, mask);
    return feature_points;
}

void FeaturePointsManager::extract_new_feature_points(cv::Mat img) {
    cv::Mat mask = compute_mask(img.rows, img.cols);
    int iter = 0;
    float weight = 1.0f;
    while (!is_enough_points() && iter < 4) {
        iter++;

        std::vector<cv::Point2f> additional_feature_points = extract_feature_points(img, mask, weight);
        std::cout << "add new feature points, num : " << additional_feature_points.size() << '\n';

        feature_points_ += additional_feature_points;
        for (cv::Point2f point : additional_feature_points) {
            put_val_around(0, mask, point.x, point.y, 3, 3);
        }

        weight *= 0.8;
    }

    if (feature_points_.size() < 4) {
        std::cout << "do not have enough points for tracking, programm exited." << std::endl;
        std::exit(0);
    }
}

cv::Mat FeaturePointsManager::compute_mask(int rows, int cols) {
    cv::Mat mask = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
	mask(bbox_.window_).setTo(255);
    return mask;
}

void FeaturePointsManager::process_feature_points(cv::Mat img,
                                                  const std::vector<cv::Point2f>& feature_points_at_new_position,
                                                  std::vector<uchar>& status) {
    visualize(img, feature_points_at_new_position);
    std::vector<cv::Vec2f> motion = compute_pixel_motion(this->feature_points_, feature_points_at_new_position);
    update_status(motion, status);
    update_bbox(motion, status, img.cols, img.rows);
    update_feature_points(feature_points_at_new_position, status);

    extract_new_feature_points(img);
}

void FeaturePointsManager::visualize(cv::Mat img, const std::vector<cv::Point2f>& feature_points_at_new_position) {
    cv::Mat vis;
    cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);

    draw_points(vis, feature_points_, cv::Scalar(255, 0, 0));
    draw_arrowed_lines(vis, feature_points_, feature_points_at_new_position, cv::Scalar(0, 0, 255), 1);

    auto tl = bbox_.top_left();
    draw_bounding_box_vis_image(vis, tl.x - 0.05 * bbox_.width(), tl.y - 0.05 * bbox_.height(), 1.1f * bbox_.width(),
                                1.1f * bbox_.height());

    cv::imshow("Optical flow tracker", vis);
    cv::waitKey(1);
}

void FeaturePointsManager::update_status(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status) {
    bool all_tracking_lost = true;
    for (uchar s : status) {
        if (s) {
            all_tracking_lost = false;
            break;
        }
    }

    if (all_tracking_lost) {
        std::cout << "all points are not valid after optical flow tracking, progrom finished" << std::endl;
        std::exit(0);
    }

    mark_status_with_contained_points(status);
    mark_status_with_amplitude(motion, status, 1.2);
    mark_status_with_angle(motion, status, 25);
}

void FeaturePointsManager::mark_status_with_contained_points(std::vector<uchar>& status) {
    // change the status if points are out of the boundingbox
	for (int i = 0; i < feature_points_.size(); i++)
	{
		if (!status[i]) //feature points is failed
			continue;
		if (!bbox_.contains(feature_points_[i]))//points are out of the boundingbox， not tracking
			status[i] = 0;
	}
}

void FeaturePointsManager::mark_status_with_amplitude(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status,
                                                      float ratio) {
    // change the status if the amplitude of the motion is outlier
	std::vector<float> amplitudes;
	for (auto & m : motion)
	{
		float amplitude = (float)std::sqrt(std::pow(m(0), 2) + std::pow(m(1), 2));
		amplitudes.push_back(amplitude);
	}
	std::nth_element(amplitudes.begin(), amplitudes.begin()+ amplitudes.size() / 2, amplitudes.end());
	float med_amplitude = *(amplitudes.begin() + amplitudes.size() / 2);
	//float med_amplitude = std::accumulate(amplitudes.begin(), amplitudes.end(), float(0) / amplitudes.size());
	for (int i = 0; i < amplitudes.size(); i++)
	{
		if (!status[i]) continue;
		if (amplitudes[i] > med_amplitude * ratio)
			status[i] = 0;
	}
}

void FeaturePointsManager::mark_status_with_angle(const std::vector<cv::Vec2f>& motion, std::vector<uchar>& status,
                                                  float shift) {
    // change the status if the orientation of the motion is outlier, 角度的判断存在问题
	std::vector<float> angles;
	for (auto & m : motion)
	{
		float angle = (float)std::atan2(m(1), m(0))*180/CV_PI;
		angles.push_back(angle);
	}
	std::vector<float> dists;
	for (int i = 0; i < angles.size(); i++)
	{
		std::vector<float>per_dists;
		for (int j = 0; j < angles.size(); j++)
		{
			if (j == i) { continue; }
			if (std::abs(angles[j] - angles[i]) < 180)
				per_dists.push_back(std::abs(angles[j] - angles[i]));
			else if (angles[j] - angles[i] < -180)
				per_dists.push_back(std::abs(angles[j] - angles[i] + 360));
			else if (angles[j] - angles[i] > 180)
				per_dists.push_back(std::abs(angles[j] - angles[i] - 360));
		}
		std::sort(per_dists.begin(), per_dists.end());
		float med_angle = per_dists[std::round(0.5* per_dists.size())];
		dists.push_back(med_angle);
	}
	//for (int i = 0; i < dists.size(); i++)
	//{
	//	if (dists[i] > shift)
	//		status[i] = 0;
	//}
}

void FeaturePointsManager::update_bbox(const std::vector<cv::Vec2f>& motions, std::vector<uchar>& status, int width_img,
                                       int height_img) {
    cv::Vec2f delta_motion;
    // todo compute motion of the bbox according to motion of feature points
	cv::Vec2f sum(0, 0);
	int cnt = 0;
	for (int i = 0; i < status.size(); ++i)
	{
		if (!status[i])continue;
		cnt++;
		sum += motions[i];
	}
	delta_motion = sum / cnt;
    bbox_.move(delta_motion[0], delta_motion[1]);

    // check if the bbox move out of image area
    cv::Rect2i rect_img(0, 0, width_img, height_img);
    cv::Rect2i intersection =
        get_intersection_from_ul(rect_img, bbox_.top_left().x, bbox_.top_left().y, bbox_.width(), bbox_.height());

    if (intersection.area() < 200) {
        std::cout << "bounding box move out of range, tracking finished.";
        std::exit(0);
    }
}

void FeaturePointsManager::update_feature_points(const std::vector<cv::Point2f>& feature_points_at_new_position,
                                                 std::vector<uchar>& status) {
    assert(feature_points_at_new_position.size() == status.size());
    // update feature points via status(give up bad points)
	feature_points_.clear();
	for (int i = 0; i < feature_points_at_new_position.size(); i++)
	{
		if (status[i])
			feature_points_.push_back(feature_points_at_new_position[i]);
	}
}

std::vector<cv::Vec2f> compute_pixel_motion(const std::vector<cv::Point2f>& old_feature_points,
                                            const std::vector<cv::Point2f>& new_feature_points) {
    std::vector<cv::Vec2f> result(old_feature_points.size());
    std::transform(old_feature_points.begin(), old_feature_points.end(), new_feature_points.begin(), result.begin(),
                   [=](cv::Point2f old_feature_point, cv::Point2f new_feature_point) {
                       return cv::Vec2f(new_feature_point.x - old_feature_point.x,
                                        new_feature_point.y - old_feature_point.y);
                   });
    return result;
}
