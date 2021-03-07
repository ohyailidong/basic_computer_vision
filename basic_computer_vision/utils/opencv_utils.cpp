#include "opencv_utils.h"
#include "opencv2/imgproc.hpp"

cv::Mat read_img(std::string img_path, cv::ImreadModes read_mode) {
    cv::Mat img = cv::imread(img_path, read_mode);
    assert(!img.empty());
    return img;
}

std::vector<cv::Mat> record_webcam() {
    std::vector<cv::Mat> result;
    cv::VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(0)) return {};
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;  // end of video stream
        cv::imshow("this is you, smile! :)", frame);
        auto key = cv::waitKey(10);
        if (key == 27) {
            break;  // stop capturing by pressing ESC
        } else {
            result.push_back(frame);
        }
    }
    return result;
}

cv::Point2i template_matching(cv::Mat img, cv::Mat temp) {
    assert(!img.empty() && !temp.empty());
    cv::Mat result;
    cv::matchTemplate(img, temp, result, cv::TM_CCORR_NORMED);

    double val_min, val_max;
    cv::Point ul_min, ul_max;
    cv::minMaxLoc(result, &val_min, &val_max, &ul_min, &ul_max, cv::Mat());
    return ul_max;
}

cv::Rect get_intersection_from_ul(cv::Rect rect_img, int x, int y, int width, int height) {
    cv::Rect roi = cv::Rect(cv::Point(x, y), cv::Size(width, height));
    cv::Rect intersection = rect_img & roi;
    return intersection;
}

cv::Rect get_intersection_from_ul(cv::Mat image, int x, int y, int width, int height) {
    cv::Rect img_rect = cv::Rect(cv::Point(0, 0), image.size());
    return get_intersection_from_ul(img_rect, x, y, width, height);
}

cv::Rect get_intersection_around(cv::Mat image, int x, int y, int width, int height) {
    return get_intersection_from_ul(image, x - width / 2, y - height / 2, width, height);
}

cv::Mat get_sub_image_around(cv::Mat image, int x, int y, int width, int height) {
    cv::Rect intersection = get_intersection_around(image, x, y, width, height);
    cv::Mat sub_img = cv::Mat::zeros(intersection.size(), image.type());
    image(intersection).copyTo(sub_img);
    return sub_img;
}

cv::Mat get_sub_image_from_ul(cv::Mat image, int x, int y, int width, int height) {
    cv::Rect intersection = get_intersection_from_ul(image, x, y, width, height);
    cv::Mat sub_img = cv::Mat::zeros(intersection.size(), image.type());
    image(intersection).copyTo(sub_img);
    return sub_img;
}
cv::Mat draw_bounding_box_vis_image(cv::Mat image, float x_ul, float y_ul, float width, float height) {
    cv::rectangle(image, cv::Rect2i(x_ul, y_ul, width, height), cv::Scalar(0, 255, 0), 2);
    return image;
}

bool is_good_mat(cv::Mat mat, std::string mat_name) {
    if (mat.empty()) {
        std::cerr << "mat " << mat_name << " is empty!\n";
        return false;
    }

    cv::Mat mask = cv::Mat(mat != mat);
    if (cv::sum(mask)[0]) {
        std::cerr << "mat " << mat_name << " has inf or nan\n";
        return false;
    }

    return true;
}