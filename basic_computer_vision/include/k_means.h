#include "../common_define.h"

enum CENTER_METHOD {
	RANDOM_CENTER =0,
	K_MEANS_PP_CENTER =1,
	PLUS =2,
};
struct Sample {
    Sample(const std::array<float, 3>& feature, const int row, const int col,
           const int label = -1)
        : feature_(feature), row_(row), col_(col), label_(label){};

    std::array<float, 3> feature_;  // feature vector of the data
    int label_;                     // corresponding center id
    int row_;                       // row in original image
    int col_;                       // col in original image
};

struct Center {
    std::array<float, 3> feature_;  // center's position
};


class Kmeans {
   public:
    Kmeans(cv::Mat img, const int k, int method=0);

    std::vector<Sample> get_result_samples() const;
    std::vector<Center> get_result_centers() const;
    void run(int max_iteration, float smallest_convergence_rate);

   private:
    void initialize_centers();
	void initialize_centers_pp();
	void initialize_centers_plus();
    void update_labels();
    void update_centers();
    bool is_terminate(int current_iter, int max_iteration,
                      float smallest_convergence_rate) const;

	int method_center;
    std::vector<Sample> samples_;       // all the data to be clustered
    std::vector<Center> centers_;       // center of each cluster
    std::vector<Center> last_centers_;  // center of last iteration
};
