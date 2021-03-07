#pragma once
#include <opencv2/core/core.hpp>
class BoundingBox {
public:
	BoundingBox(float x, float y, float width, float height) : window_(x, y, width, height) {};
	BoundingBox() = default;

	void move(float delta_x, float delta_y) {
		window_.x += delta_x;
		window_.y += delta_y;
	}

	void move_top_left_to(float x, float y) {
		window_.x = x;
		window_.y = y;
	}


	void move_center_to(float x, float y) {
		return move_top_left_to(x - window_.width / 2.f, y - window_.height / 2.f);
	}

	const cv::Point2f top_left() const {
		return window_.tl();
	}
	const cv::Point2f bottom_right() const {
		return window_.br();
	}

	int area() const {
		return window_.width * window_.height;
	}

	int width() const {
		return window_.width;
	}

	int height() const {
		return window_.height;
	}

	const cv::Point2f center() const {
		return cv::Point2f(window_.tl().x + window_.width / 2, window_.tl().y + window_.height / 2);
	}

	bool contains(cv::Point2f point) const {
		return window_.contains(point);
	}

public:
	cv::Rect2f window_;
};
