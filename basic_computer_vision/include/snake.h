#pragma once
#include "gradient_descent_base.h"

class Contour 
{
   public:
    Contour(int max_x, int max_y, double radius, cv::Point2d center,
            int num_points);
    Contour(cv::Mat points);
    Contour(const Contour& contour);
    Contour& operator=(const Contour& contour);

    cv::Vec2d& operator[](int index);

    cv::Mat get_points() const;
    int get_num_points() const;

   private:
    cv::Mat points_;
};

struct ParamSnake 
{
    ParamSnake(double alpha, double beta, double step_size);
    double alpha_;
    double beta_;

    double step_size_;
};


class Snake : public GradientDescentBase 
{
   public:
    Snake(cv::Mat original_img, cv::Mat gvf_x, cv::Mat gvf_y, Contour contour,
          ParamSnake param_snake);

    Contour get_contour() const;

   private:
    void initialize() override;
    void update() override;

    void cal_internal_force_matrix();

    std::string return_drive_class_name() const override;
    void roll_back_state() override;
    void back_up_state() override;
    void print_terminate_info() const override;
    double compute_energy() override;

    cv::Mat original_img_;           // original image for visualization
    cv::Mat internal_force_matrix_;  // (Id-(alpha*A - beta*B))
    ParamSnake param_snake_;
    Contour contour_;       // current contour
    Contour last_contour_;  // contour of last step

    cv::Mat gvf_x_;        // gvf in the x direction
    cv::Mat gvf_y_;        // gvf in the y direction
    cv::Mat gvf_contour_;  // gvf on the contour

	cv::Mat deriv_1, deriv_3, A, B;
};
