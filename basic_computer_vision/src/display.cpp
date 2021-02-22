
#include "..\include\display.h"

void draw_optical_flow(cv::Mat& fx, cv::Mat& fy, cv::Mat& cflowmap, int step,
                       double scaleFactor, cv::Scalar& color) {
    for (int r = 0; r < cflowmap.rows; r += step)
        for (int c = 0; c < cflowmap.cols; c += step) {
            cv::Point2f fxy;

            fxy.x = fx.at<double>(r, c);
            fxy.y = fy.at<double>(r, c);

            if (fxy.x != 0 || fxy.y != 0) {
                cv::line(cflowmap, cv::Point(c, r),
                         cv::Point(cvRound(c + (fxy.x) * scaleFactor),
                                   cvRound(r + (fxy.y) * scaleFactor)),
                         color, 1, cv::LINE_AA);
            }
            cv::circle(cflowmap, cv::Point(c, r), 1, cv::Scalar(255, 0, 0), 1);
        }
}

void display_gvf(cv::Mat fx, cv::Mat fy, int delay, bool save = false) {
    cv::Mat cflowmap = cv::Mat::zeros(fx.size(), CV_8UC3);

    int step = 8;
    double scaleFactor = 7;
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Mat disp_fx = fx.clone();
    cv::Mat disp_fy = fy.clone();
    cv::normalize(disp_fx, disp_fx, -1, 1, cv::NORM_MINMAX);
    cv::normalize(disp_fy, disp_fy, -1, 1, cv::NORM_MINMAX);
    draw_optical_flow(disp_fx, disp_fy, cflowmap, step, scaleFactor, color);
    disp_image(cflowmap, "gvf display", delay);
    if (save) cv::imwrite("gvf_display.png", cflowmap);
}

//--Overloaded functions to display an image in a new window--//
void disp_image(cv::Mat& img) {
    if (img.empty()) { 
        std::cout << "Error reading image file!";
        std::cin.ignore();
    } else {
        cv::namedWindow("Image", 0);
        cv::imshow("Image", img);
        cv::waitKey();
    }
}

void disp_image(cv::Mat& img, cv::String windowName) {
    if (img.empty()) {                         
        std::cout << "Error reading image File!";
        std::cin.ignore();
    } else {
        cv::imshow(windowName, img);
        cv::waitKey();
    }
}

void disp_image(cv::Mat& img, cv::String windowName, int delay) {
    if (img.empty()) { 
        std::cout << "Error reading image File!";
        std::cin.ignore();
    } else {
        cv::namedWindow(windowName, 0);
        cv::imshow(windowName, img);
        cv::waitKey(delay);
    }
}

void disp_image(cv::Mat& img, cv::String windowName, cv::String error_msg) {
    if (img.empty()) { 
        std::cout << error_msg;
        std::cin.ignore();
    } else {
        cv::namedWindow(windowName, 0);
        cv::imshow(windowName, img);
        cv::waitKey();
    }
}

void disp_image(cv::Mat& img, cv::String windowName, cv::String error_msg,
                int delay) {
    if (img.empty()) { 
        std::cout << error_msg;
        std::cin.ignore();
    } else {
        cv::namedWindow(windowName, 0);
        cv::imshow(windowName, img);
        cv::waitKey(delay);
    }
}

void display_contour(cv::Mat img, Contour& contour, int delay) {
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < contour.get_num_points() - 1; i++) {
        cv::line(img_rgb, cv::Point2d(contour[i]), cv::Point2d(contour[i + 1]),
                 cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
    }
    cv::line(img_rgb, cv::Point2d(contour[0]),
             cv::Point2d(contour[contour.get_num_points() - 1]),
             cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
    cv::imshow("snake", img_rgb);
    cv::waitKey(delay);
}