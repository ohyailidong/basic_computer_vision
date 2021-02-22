#pragma once
#include "snake.h"

#include "..\common_define.h"

void disp_image(cv::Mat& img, cv::String windowName);
void disp_image(cv::Mat& img, cv::String windowName, int delay);
void draw_optical_flow(cv::Mat& fx, cv::Mat& fy, cv::Mat& cflowmap, int step,
                       double scaleFactor, cv::Scalar& color);
void display_gvf(cv::Mat fx, cv::Mat fy, int delay, bool save);
void display_contour(cv::Mat img, Contour& contour, int delay);
