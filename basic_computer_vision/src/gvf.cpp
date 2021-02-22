#include "..\include\gvf.h"
#include "..\include\display.h"
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>


ParamGVF::ParamGVF(double smooth_term_weight, double init_step_size)
    : smooth_term_weight_(smooth_term_weight), init_step_size_(init_step_size) {
}


GVF::GVF(cv::Mat grad_original_x, cv::Mat grad_original_y,
         const ParamGVF& param_gvf)
    : GradientDescentBase(param_gvf.init_step_size_),
      param_gvf_(param_gvf),
      data_term_weight_(cv::Mat::zeros(grad_original_x.size(), CV_64F)),
      laplacian_gvf_x_(cv::Mat::zeros(grad_original_x.size(), CV_64F)),
      laplacian_gvf_y_(cv::Mat::zeros(grad_original_y.size(), CV_64F))
{
    // initialize the gvf external energy
    cv::Mat square_grad_original_x, square_grad_original_y;
    cv::pow(grad_original_x, 2.0f, square_grad_original_x);
    cv::pow(grad_original_y, 2.0f, square_grad_original_y);

    cv::Mat mag_original;
    cv::sqrt(square_grad_original_x + square_grad_original_y, mag_original);
    cv::GaussianBlur(mag_original, mag_original, cv::Size(3, 3), 3, 3);
    cv::Sobel(mag_original, gvf_initial_x_, CV_64F, 1, 0, 3);
    cv::Sobel(mag_original, gvf_initial_y_, CV_64F, 0, 1, 3);

    // compute the date term weight
    cv::Mat square_gvf_initial_x, square_gvf_initial_y;
    cv::pow(gvf_initial_x_, 2.0f, square_gvf_initial_x);
    cv::pow(gvf_initial_y_, 2.0f, square_gvf_initial_y);
    data_term_weight_ = square_gvf_initial_x + square_gvf_initial_y;
}

void GVF::initialize() {
    gvf_x_ = gvf_initial_x_.clone();
    gvf_y_ = gvf_initial_y_.clone();
}


void GVF::update() {
	back_up_state();
	// calculate Laplacian
	cv::Laplacian(last_gvf_x_, laplacian_gvf_x_, CV_64F);
	cv::Laplacian(last_gvf_y_, laplacian_gvf_y_, CV_64F);
	//update gvf_x_,gvf_y_
	gvf_x_ = last_gvf_x_ + param_gvf_.init_step_size_*(param_gvf_.smooth_term_weight_* laplacian_gvf_x_ -
		(last_gvf_x_ - gvf_initial_x_).mul(data_term_weight_));
	gvf_y_ = last_gvf_y_ + param_gvf_.init_step_size_*(param_gvf_.smooth_term_weight_* laplacian_gvf_y_ -
		(last_gvf_y_ - gvf_initial_y_).mul(data_term_weight_));

    display_gvf(gvf_x_, gvf_y_, 1, false);
}
/**
 * @brief compute enegy according to current gvf
 *
 * @return double
 */
double GVF::compute_energy() {
	// compute data term energy
    float smooth_term_energy =0.f;
	cv::Mat u_x, u_x_square, u_y, u_y_square;
	cv::Sobel(gvf_x_, u_x, CV_64F, 1, 0, 3);//ux
	cv::Sobel(gvf_x_, u_y, CV_64F, 0, 1, 3);//uy
	cv::Mat v_x, v_x_square, v_y, v_y_square;
	cv::Sobel(gvf_y_, v_x, CV_64F, 1, 0, 3);//vx
	cv::Sobel(gvf_y_, v_y, CV_64F, 0, 1, 3);//vy
	cv::pow(u_x, 2.0f, u_x_square);
	cv::pow(u_y, 2.0f, u_y_square);
	cv::pow(v_x, 2.0f, v_x_square);
	cv::pow(v_y, 2.0f, v_y_square);
	cv::Mat sum = param_gvf_.smooth_term_weight_*(u_x_square + u_y_square + v_x_square + v_y_square);
	for (int i = 0; i < sum.rows; ++i)
		for (int j = 0; j < sum.cols; ++j)
			smooth_term_energy += sum.at<double>(i, j);
    // compute smooth term energy
    double data_term_energy = 0.f;
	double deltaX = 0.f;
	double deltaY = 0.f;
	double deltaF = 0.f;
	for (int i = 0; i < gvf_x_.rows; ++i)
	{
		for (int j = 0; j < gvf_x_.cols; ++j)
		{
			deltaX = gvf_x_.at<double>(i, j) - gvf_initial_x_.at<double>(i, j);
			deltaY = gvf_y_.at<double>(i, j) - gvf_initial_y_.at<double>(i, j);
			deltaF = data_term_weight_.at<double>(i, j);
			data_term_energy += deltaF * (deltaX*deltaX + deltaY * deltaY);
		}
	}

    return smooth_term_energy + data_term_energy;
}


void GVF::roll_back_state() {
    gvf_x_ = last_gvf_x_;
    gvf_y_ = last_gvf_y_;
}

void GVF::back_up_state() {
    last_gvf_x_ = gvf_x_.clone();
    last_gvf_y_ = gvf_y_.clone();
}

std::vector<cv::Mat> GVF::get_result_gvf() const {
    std::vector<cv::Mat> gvf_result(2);
    gvf_result[0] = gvf_x_.clone();
    gvf_result[1] = gvf_y_.clone();
    return gvf_result;
}

void GVF::print_terminate_info() const {
    std::cout << "GVF iteration finished." << std::endl;
}

std::string GVF::return_drive_class_name() const {
    return "GVF";
}