#pragma once

#include "../include/ms_segmentation.h"
#include "../utils/opencv_utils.h"
#include "../utils/visualizer_3d.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>

class TestMeanShift_Seg
{
public:
	inline static void Run();
};
void TestMeanShift_Seg::Run() {
	const std::string imageFileName = "images/test_data/diesel.jpeg";
	cv::Mat img = read_img(imageFileName, cv::IMREAD_COLOR);

	cv::resize(img, img, cv::Size(100, 100));

	Vis3D* visualizer_ptr = new Vis3D();
	MeanShiftSeg mss(50, visualizer_ptr);  // mss take the ownership of visualizer_ptr;

	std::thread vis_thread(&Vis3D::visualize, std::ref(*visualizer_ptr));
	mss.process(img);

	vis_thread.join();
}
