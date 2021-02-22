#include "..\include\display.h"
#include "..\include\gvf.h"
#include "..\include\snake.h"

class Test_GVF_Snake
{
public:
	static void Run();
};
void Test_GVF_Snake::Run()
{
	const std::string filename = "images/gvf_display.png";

    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // preprocess the input image
    cv::GaussianBlur(img, img, cv::Size(3, 3), 3, 3);

    cv::Mat grad_original_x, grad_original_y;
    cv::Sobel(img, grad_original_x, CV_64F, 1, 0, 3);
    cv::Sobel(img, grad_original_y, CV_64F, 0, 1, 3);

    // run gvf at first to get the gradient vecotor flow
    double smooth_term = 1e8;
    double step_size = 2e-9;
    ParamGVF param_gvf(smooth_term, step_size);
    GVF gvf(grad_original_x, grad_original_y, param_gvf);

    int max_iteration_gvf = 1e4;
    gvf.run(max_iteration_gvf);  // parameter: max_iteration
    std::vector<cv::Mat> gvf_result = gvf.get_result_gvf();

    display_gvf(gvf_result[0], gvf_result[1], 0, true);
    std::cout << gvf_result[0].type() << std::endl;

    // run snake
    int max_x = gvf_result[0].rows;
    int max_y = gvf_result[1].cols;
    double radius = std::min(max_x, max_y) / 3.5f;
    cv::Point2d center(max_x / 2.f, max_y / 2.f);
    int num_points = 200;

    Contour contour(max_x, max_y, radius, center, num_points);
    ParamSnake param_snake(0, 0, 1e-10);

    Snake snake_model(img, gvf_result[0], gvf_result[1], contour, param_snake);
    snake_model.run(1000);
    Contour result_contour = snake_model.get_contour();
    //display_contour(img, result_contour, 0);

}