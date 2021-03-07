#pragma once

#include "..\include\bounding_box.h"
#include "..\utils\opencv_utils.h"
#include "..\include\optical_flow_tracker.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TestOpticalFlow
{
public:
	inline static void Run();
};
void TestOpticalFlow::Run() {
	const std::string videoFileName = "images/dataset/Car2/";
	const std::string templateFileName = "images/dataset/Car2/template/template.jpg";

    std::vector<cv::Mat> video;
    for (int id = 1; id < 875; id++) {
        cv::Mat img = cv::imread(videoFileName + std::to_string(id) + ".jpg", cv::IMREAD_GRAYSCALE);
        assert(!img.empty());
        video.push_back(img);
    }

    // detect object in the first image with template
    cv::Mat temp = cv::imread(templateFileName, cv::IMREAD_GRAYSCALE);
    cv::Point2i center = template_matching(video[0], temp);

    float w = 0.5;
    BoundingBox bbox_init(center.x , center.y, 
		temp.cols, temp.rows);
    OpticalFlowTracker tracker;
    tracker.process(bbox_init, video);
}