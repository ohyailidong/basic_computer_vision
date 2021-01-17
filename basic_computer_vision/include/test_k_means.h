#include "../include/k_means.h"

#define CLUSTER_NUM    3    //分类cluster数目
#define MAXITER_NUM 1000    //最大迭代次数
#define INIT_CENTER 2		//初始化中心点方法

const std::string filename0 = "images/test_data/lena.png";
const std::string filename1 = "images/test_data/building.jpeg";
const std::string filename2 = "images/test_data/trump.jpg";

class Test_Kmeans
{
public: 
	static void Run();
};
void Test_Kmeans::Run()
{
	cv::Mat img = cv::imread(filename0, 1);

	if (img.empty() || img.channels() != 3)
		std::exit(-1);

	int convergence_radius = 1e-6;//最小收敛半径

	Kmeans kmeans(img, CLUSTER_NUM, INIT_CENTER);
	kmeans.run(MAXITER_NUM, convergence_radius);

	std::vector<Sample> samples = kmeans.get_result_samples();
	std::vector<Center> centers = kmeans.get_result_centers();

	cv::Mat result(img.size(), img.type());

	for (const Sample& sample : samples) {
		for (int channel = 0; channel < 3; channel++) {
			result.at<cv::Vec3b>(sample.row_, sample.col_)[channel] =
				centers[sample.label_].feature_[channel];
		}
	}
	cv::Mat concat_img;
	cv::hconcat(img, result, concat_img);
	cv::imshow("left: original image, right: kmeans result", concat_img);
	cv::waitKey(0);
}