#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

class HeightMap;

void disp_image(cv::Mat& img, cv::String windowName);
void disp_image(cv::Mat& img, cv::String windowName, int delay);
void draw_optical_flow(cv::Mat& fx, cv::Mat& fy, cv::Mat& cflowmap, int step, double scaleFactor, cv::Scalar& color);
// void display_gvf(cv::Mat fx, cv::Mat fy, int delay, bool save);
// void display_contour(cv::Mat img, Contour& contour, int delay);

/**
 * @brief Map image in a coloful space and draw a contour
 *
 * @param sdf_map
 * @return cv::Mat
 */
cv::Mat draw_height_map(const HeightMap& height_map);
cv::Mat apply_jetmap(cv::Mat image);
cv::Mat draw_points(cv::Mat img, cv::Mat points, cv::Scalar color);
/**
 * @brief get a cv::Mat of type float or double
 * return a normlized image for visualization;
 *
 * @return cv::Mat
 */
cv::Mat get_float_mat_vis_img(cv::Mat);
void display_and_drag_to_print_pixel_value_8UC3(cv::Mat img);

void display_float_mat_img(cv::Mat img, int delay, std::string win_name = " ");