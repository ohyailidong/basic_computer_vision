#include "..\include\k_means.h"


// to generate random number
static std::random_device rd;
static std::mt19937 rng(rd());

/**
 * @brief get_random_index, check_convergence, calc_square_distance are helper
 * functions, you can use it to finish your homework:)
 *
 */

std::set<int> get_random_index(int max_idx, int n);

float check_convergence(const std::vector<Center>& current_centers,
                        const std::vector<Center>& last_centers);

inline float calc_square_distance(const std::array<float, 3>& arr1,
                                  const std::array<float, 3>& arr2);

/**
 * @brief Construct a new Kmeans object
 *
 * @param img : image with 3 channels
 * @param k : wanted number of cluster
 */
Kmeans::Kmeans(cv::Mat img, const int k) {
    centers_.resize(k);
    last_centers_.resize(k);
    samples_.reserve(img.rows * img.cols);

    // save each feature vector into samples
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            std::array<float, 3> tmp_feature;
            for (int channel = 0; channel < 3; channel++) {
                tmp_feature[channel] =
                    static_cast<float>(img.at<cv::Vec3b>(r, c)[channel]);
            }
            samples_.emplace_back(tmp_feature, r, c, -1);
        }
    }
}

/**
 * @brief initialize k centers randomly, using set to ensure there are no
 * repeated elements
 *
 */
// TODO Try to implement a better initialization function
void Kmeans::initialize_centers() {
    std::set<int> random_idx =
        get_random_index(samples_.size() - 1, centers_.size());
    int i_center = 0;

    for (auto index : random_idx) {
        centers_[i_center].feature_ = samples_[index].feature_;
        i_center++;
    }
}
//Kmeans++ 方法实现初始化
void Kmeans::initialize_centers_pp()
{
	auto random_idx= get_random_index(samples_.size() - 1, 1);
	int index = *random_idx.begin();
	//初始化第一个中心点
	centers_[0].feature_ = samples_[index].feature_;
	//计算每个采样点到中心点的距离
	double sum0 = 0;
	const int N = samples_.size();
	float* dist = new float[N];

	for (int i = 0; i < N; i++)
	{
		dist[i] = calc_square_distance(samples_[i].feature_, centers_[0].feature_);
		sum0 += dist[i];
	}
	//计算其他中心点
	for (int k = 1; k < centers_.size(); k++)
	{
		double ratio= rand() % (1000) / (float)(1000);
		double p = ratio *sum0;
		int ci = 0;
		for (; ci < N - 1; ci++)
		{
			p -= dist[ci];
			if (p <= 0)
				break;
		}
		//取样本点到中心点的最小距离
		double s = 0.;
		centers_[k].feature_ = samples_[ci].feature_;
		for (int i = 0; i < N; i++)
		{
			dist[i] = std::min(calc_square_distance(samples_[i].feature_, centers_[k].feature_), dist[i]);
			s += dist[i];
		}
		sum0 = s;
	}
	delete[] dist;
}
/**
 * @brief change the label of each sample to the nearst center
 *
 */
void Kmeans::update_labels() {
    for (Sample& sample : samples_) {
        // TODO update labels of each feature
		int k_best = 0;
		double min_dist = DBL_MAX;
		for (int i = 0; i < centers_.size(); i++)
		{
			float tempDist = calc_square_distance(sample.feature_, centers_[i].feature_);
			if (tempDist < min_dist)
			{
				min_dist = tempDist;
				k_best = i;
			}
		}
		sample.label_ = k_best;
    }
}

/**
 * @brief move the centers according to new lables
 *
 */
void Kmeans::update_centers() {
    // backup centers of last iteration
    last_centers_ = centers_;
    // calculate the mean value of feature vectors in each cluster
	for (auto &center : centers_)
		center = { 0,0,0 };
	std::vector<int>labelcount(centers_.size(),0);
	for (int i = 0; i < samples_.size(); i++)
	{
		int k = samples_[i].label_;
		for(int j=0;j<3;j++)
			centers_[k].feature_[j] += samples_[i].feature_[j];
		labelcount[k]++;
	}
	// if some cluster appeared to be empty then:
	//   1. find the biggest cluster
	//   2. find the farthest from the center point in the biggest cluster
	//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
	for (int k = 0; k < centers_.size(); k++)
	{
		if (labelcount[k] != 0)
			continue;
		int max_k = 0;
		for (int k1 = 1; k1 < centers_.size(); k1++)
		{
			if (labelcount[max_k] < labelcount[k1])
				max_k = k1;
		}

		double max_dist = 0;
		int farthest_i = -1;
		Center base_center = centers_[max_k];
		Center _base_center = {0,0,0}; 
		float scale = 1.f / labelcount[max_k];
		for (int j = 0; j < 3; j++)
			_base_center.feature_[j] = base_center.feature_[j] * scale;

		for (int i = 0; i < samples_.size(); i++)
		{
			if (samples_[i].label_ != max_k)
				continue;
			double dist = calc_square_distance(samples_[i].feature_, _base_center.feature_);

			if (max_dist <= dist)
			{
				max_dist = dist;
				farthest_i = i;
			}
		}

		labelcount[max_k]--;
		labelcount[k]++;
		samples_[farthest_i].label_ = k;

		for (int j = 0; j < 3; j++)
		{
			centers_[max_k].feature_[j] -= samples_[farthest_i].feature_[j];
			centers_[k].feature_[j] += samples_[farthest_i].feature_[j];
		}
	}

	for (int k = 0; k < centers_.size(); k++)
	{
		float scale = 1.f / labelcount[k];
		for (int j = 0; j < 3; j++)
			centers_[k].feature_[j] *= scale;
	}
}

/**
 * @brief check terminate conditions, namely maximal iteration is reached or it
 * convergents
 *
 * @param current_iter
 * @param max_iteration
 * @param smallest_convergence_radius
 * @return true
 * @return false
 */
bool Kmeans::is_terminate(int current_iter, int max_iteration,
                          float smallest_convergence_radius) const {
	if (max_iteration < 1)
		max_iteration = 1;
	if(current_iter>0 && (current_iter==max_iteration ||
		check_convergence(centers_, last_centers_)<= smallest_convergence_radius))
		return true;
	return false;
}

std::vector<Sample> Kmeans::get_result_samples() const {
    return samples_;
}
std::vector<Center> Kmeans::get_result_centers() const {
    return centers_;
}
/**
 * @brief Execute k means algorithm
 *                1. initialize k centers randomly
 *                2. assign each feature to the corresponding centers
 *                3. calculate new centers
 *                4. check terminate condition, if it is not fulfilled, return
 *                   to step 2
 * @param max_iteration
 * @param smallest_convergence_radius
 */
void Kmeans::run(int max_iteration, float smallest_convergence_radius) {
	initialize_centers_pp();
    initialize_centers();
    int current_iter = 0;
    while (!is_terminate(current_iter, max_iteration,
                         smallest_convergence_radius)) {
        current_iter++;
        update_labels();
        update_centers();
    }
}

/**
 * @brief Get n random numbers from 1 to parameter max_idx
 *
 * @param max_idx
 * @param n
 * @return std::set<int> A set of random numbers, which has n elements
 */
std::set<int> get_random_index(int max_idx, int n) {
    std::uniform_int_distribution<int> dist(1, max_idx + 1);

    std::set<int> random_idx;
    while (random_idx.size() < n) {
        random_idx.insert(dist(rng) - 1);
    }
    return random_idx;
}
/**
 * @brief Calculate the L2 norm of current centers and last centers
 *
 * @param current_centers current assigned centers with 3 channels
 * @param last_centers  last assigned centers with 3 channels
 * @return float
 */
float check_convergence(const std::vector<Center>& current_centers,
                        const std::vector<Center>& last_centers) {
    float convergence_radius = 0;
    for (int i_center = 0; i_center < current_centers.size(); i_center++) {
        convergence_radius +=
            calc_square_distance(current_centers[i_center].feature_,
                                 last_centers[i_center].feature_);
    }
    return convergence_radius;
}


inline float calc_square_distance(const std::array<float, 3>& arr1,
                                  const std::array<float, 3>& arr2) {
    return std::pow((arr1[0] - arr2[0]), 2) + std::pow((arr1[1] - arr2[1]), 2) +
           std::pow((arr1[2] - arr2[2]), 2);
}