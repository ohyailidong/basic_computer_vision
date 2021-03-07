#pragma once

#include "../include/bounding_box.h"
#include "../include/ms_tracker.h"
#include "../utils/opencv_utils.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TestMeanShift_Tracking
{
public:
	inline static void Run();
};
void TestMeanShift_Tracking::Run() {
	const std::string videoFileName = "images/dataset/Car2/";
	const std::string templateFileName = "images/dataset/Car2/template/template.jpg";
	std::vector<cv::Mat> video;
	for (int id = 1; id < 875; id++) {
		cv::Mat img = read_img(videoFileName + std::to_string(id) + ".jpg", cv::IMREAD_GRAYSCALE);
		assert(!img.empty());
		video.push_back(img);
	}
	cv::Mat temp = read_img(templateFileName, cv::IMREAD_GRAYSCALE);

	cv::Point2i init_upper_left = template_matching(video.front(), temp);
	cv::Point2f init_bbox_center =
		cv::Point2f(init_upper_left.x + temp.cols / 2.0f, init_upper_left.y + temp.rows / 2.0f);

	BoundingBox init_bbox(init_upper_left.x, init_upper_left.y, temp.cols, temp.rows);

	MeanShiftTracker ms_tracker(temp, init_bbox);
	ms_tracker.process(video);
}

