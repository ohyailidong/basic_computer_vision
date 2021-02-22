#pragma once
#include "gradient_descent_base.h"

struct ParamGVF
{
    ParamGVF(double smooth_term_weight = 1e8, double init_step_size = 1e-7);
    double init_step_size_;
    double smooth_term_weight_;
};

class GVF : public GradientDescentBase 
{
   public:
    GVF(cv::Mat grad_original_x, cv::Mat grad_original_y,
        const ParamGVF& param_gvf = ParamGVF(1e8, 1e-7));

    std::vector<cv::Mat> get_result_gvf() const;

   private:
    void initialize() override;
    void update() override;

    double compute_energy() override;
    void roll_back_state() override;
    void back_up_state() override;

    void print_terminate_info() const override;
    std::string return_drive_class_name() const override;

    ParamGVF param_gvf_;

    cv::Mat data_term_weight_;  // grad_original_x_**2 + grad_original_y_**2

    cv::Mat gvf_initial_x_;  // partial derivative w.r.t. x
    cv::Mat gvf_initial_y_;  // partial derivative w.r.t. y

    cv::Mat gvf_x_;
    cv::Mat gvf_y_;

    cv::Mat last_gvf_x_;
    cv::Mat last_gvf_y_;

    cv::Mat laplacian_gvf_x_;  // Laplacian of gvf_x_
    cv::Mat laplacian_gvf_y_;  // Laplacian of gvf_y_
};
