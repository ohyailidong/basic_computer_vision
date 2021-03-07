#include "..\include\ms_segmentation.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/viz3d.hpp>

inline bool is_in_radius(cv::Vec3f center, cv::Vec3f sample, double radius_square) {
    return cv::norm(center - sample, cv::NORM_L2SQR) < radius_square;
}

bool is_convergent(cv::Mat curr, cv::Mat last) {
    if (last.empty()) return false;

    assert(curr.rows == last.rows && curr.cols == last.cols);

    for (int r = 0; r < curr.rows; r++) {
        for (int c = 0; c < curr.cols; c++) {
            if (!is_in_radius(curr.at<cv::Vec3f>(r, c), last.at<cv::Vec3f>(r, c), 0.1)) {
                return false;
            }
        }
    }
    return true;
}

void MeanShiftSeg::process(cv::Mat img) {
    assert(img.type() == CV_8UC3);

    visualizer->set_origin_img(img);

    img.convertTo(features_curr_, CV_32FC3);
    features_origin_ = features_curr_.clone();

    int it = 0;
    int max_iteration = 50;
    cv::Mat features_last;

    while (!it || (it < max_iteration && !is_convergent(features_curr_, features_last))) {
        it++;
        std::cout << "curr iteration : " << it << '\n';
        features_last = features_curr_.clone();
        update_mass_center();
    }

    if (it == max_iteration + 1) {
        std::cout << "reach max iteration, segementation is stopped \n";
    } else {
        std::cout << "segementation is convergent \n";
    }
}

void MeanShiftSeg::update_mass_center() {
    visualizer->set_features(features_curr_);
#pragma omp parallel for
    for (int r_curr = 0; r_curr < features_curr_.rows; r_curr++) {
        for (int c_curr = 0; c_curr < features_curr_.cols; c_curr++) {
            // todo update the features. Each new feature is the mass center in local window
			cv::Vec3f& feature = features_curr_.at<cv::Vec3f>(r_curr, c_curr);
			cv::Vec3f sum(0.0, 0.0);
			int cnt = 0;
			for (int r_ori = 0; r_ori < features_origin_.rows; r_ori++) {
				for (int c_ori = 0; c_ori < features_origin_.cols; c_ori++) {
					cv::Vec3f feature_origin = features_origin_.at<cv::Vec3f>(r_ori, c_ori);
					if (is_in_radius(feature, feature_origin, radius_square_)) {
						sum += feature_origin;
						cnt++;
					}
				}
			}
			feature = sum / static_cast<float>(cnt);
        }
    }
}

MeanShiftSeg::MeanShiftSeg(double radius, Vis3D* vis_ptr) : radius_square_(radius * radius), visualizer(vis_ptr) {
}
