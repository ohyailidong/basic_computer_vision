#include "..\include\display.h"
#include "..\include\gvf.h"

class TestGVF
{
public:
	static void Run();
};
void TestGVF::Run() {
	const std::string filename = "images/gvf_display.png";

    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // preprocess the input image
    cv::GaussianBlur(img, img, cv::Size(3, 3), 3, 3);

    cv::Mat grad_original_x, grad_original_y;
    cv::Sobel(img, grad_original_x, CV_64F, 1, 0, 3);
    cv::Sobel(img, grad_original_y, CV_64F, 0, 1, 3);

    double smooth_term = 2e8;
    double step_size = 5e-10;
    ParamGVF param_gvf(smooth_term, step_size);
    GVF gvf(grad_original_x, grad_original_y, param_gvf);

    int max_iteration_gvf = 2e4;
    gvf.run(max_iteration_gvf);  // parameter: max_iteration
    std::vector<cv::Mat> gvf_result = gvf.get_result_gvf();

    // display gvf result please set save as true if you want to save it
    bool save = true;
    display_gvf(gvf_result[0], gvf_result[1], 0, save);

    cv::imwrite("gvf_x_star.png", gvf_result[0]);
    cv::imwrite("gvf_y_star.png", gvf_result[1]);
}