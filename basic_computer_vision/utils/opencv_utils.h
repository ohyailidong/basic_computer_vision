#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline bool is_in_img(cv::Mat img, int row, int col) {
    return row < img.rows && row >= 0 && col < img.cols && col >= 0;
}

inline int pos_to_id(int row, int col, int step) {
    return row * step + col;
}

inline cv::Point id_to_pos(int id, int step) {
    return {id / step, id % step};
}

std::vector<cv::Mat> record_webcam();
/**
 * @brief get location of a temlate given, image and a template
 *
 * @param img
 * @param temp
 * @return cv::Point2i
 */
cv::Point2i template_matching(cv::Mat img, cv::Mat temp);
/**
 * @brief get a sub image, safe at boundary without autofilling
 *
 * @param image
 * @param x : x of window center
 * @param y : y of window center
 * @param width : widht of the window
 * @param height : height of the window
 * @return cv::Mat : sub im
 */
cv::Mat get_sub_image_around(cv::Mat image, int x, int y, int width, int height);

cv::Mat draw_bounding_box_vis_image(cv::Mat image, float x, float y, float width, float height);

cv::Rect get_intersection_around(cv::Mat image, int x, int y, int width, int height);

cv::Rect get_intersection_from_ul(cv::Mat image, int x, int y, int width, int height);
cv::Rect get_intersection_from_ul(cv::Rect rect_img, int x, int y, int width, int height);

template <typename T>
void put_val_from_ul(T val, cv::Mat input_mat, int x_ul, int y_ul, int width, int height);

template <typename T>
void put_val_around(T val, cv::Mat input_mat, int x_center, int y_center, int width, int height);

template <typename T>
void draw_points(cv::Mat img, const std::vector<cv::Point_<T>>& points, cv::Scalar bgr = cv::Scalar(0, 0, 255));

template <typename T>
void draw_lines(cv::Mat img, const std::vector<cv::Point_<T>>& src, const std::vector<cv::Point_<T>>& target,
                cv::Scalar bgr = cv::Scalar(0, 0, 255), int width = 2);

template <typename T>
void draw_arrowed_lines(cv::Mat img, const std::vector<cv::Point_<T>>& src, const std::vector<cv::Point_<T>>& target,
                        cv::Scalar bgr = cv::Scalar(0, 0, 255), int width = 2);

/*--------------------------------------------------------
#####################implementation: template function #####################
---------------------------------------------------------*/
template <typename T>
void put_val_from_ul(T val, cv::Mat input_mat, int x_ul, int y_ul, int width, int height) {
    cv::Rect intersection = get_intersection_from_ul(input_mat, x_ul, y_ul, width, height);
    input_mat(intersection) = val * cv::Mat::ones(intersection.size(), CV_8UC1);
}

template <typename T>
void put_val_around(T val, cv::Mat input_mat, int x_center, int y_center, int width, int height) {
    assert(width % 2 == 1 && height % 2 == 1);
    put_val_from_ul(val, input_mat, x_center - width / 2, y_center - height / 2, width, height);
}

template <typename T>
void draw_points(cv::Mat img, const std::vector<cv::Point_<T>>& points, cv::Scalar bgr) {
    std::for_each(points.begin(), points.end(), [&](const cv::Point_<T> point) { cv::circle(img, point, 1, bgr, 2); });
}

template <typename T>
void draw_lines(cv::Mat img, const std::vector<cv::Point_<T>>& src, const std::vector<cv::Point_<T>>& target,
                cv::Scalar bgr, int width) {
    auto it_target = target.begin();
    std::for_each(src.begin(), src.end(),
                  [&](const cv::Point_<T> point) { cv::line(img, point, *it_target++, bgr, width); });
}

template <typename T>
void draw_arrowed_lines(cv::Mat img, const std::vector<cv::Point_<T>>& src, const std::vector<cv::Point_<T>>& target,
                        cv::Scalar bgr, int width) {
    auto it_target = target.begin();
    std::for_each(src.begin(), src.end(),
                  [&](const cv::Point_<T> point) { cv::arrowedLine(img, point, *it_target++, bgr, width); });
}

cv::Mat do_sobel(cv::Mat input, int flag = 0);