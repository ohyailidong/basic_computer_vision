#include "bounding_box.h"
#include "histogram.h"
#include <opencv2/core.hpp>

cv::Mat compute_gauss_kernel(int width, int height, double sigma);

Histogram make_histogramm(cv::Mat img, int num_bins, cv::Mat weight);

class MeanShiftTracking {
   public:

    MeanShiftTracking(cv::Mat temp, int num_bin_histogramm = 16)
        : hist_temp_(num_bin_histogramm, 0, 255), bbox_(0, 0, temp.cols, temp.rows) {
        temp.convertTo(temp_64f_, CV_64FC1);
        weight_geometirc_ = compute_gauss_kernel(temp.cols, temp.rows, 10);

        hist_temp_ = make_histogramm(temp_64f_, num_bin_histogramm, weight_geometirc_);
    }
    void run(int max_iteration, cv::Point2f init_object_center, cv::Mat img_curr);

    cv::Point2f get_tracked_object_center() const {
        return bbox_.center();
    }

    void visualize(cv::Mat curr_img) const;

   private:

    bool make_sure_score_increase(const BoundingBox& bbox_before_update, cv::Mat img, double score_before_update);

    cv::Mat temp_64f_;          // template in the data type : CV_64F
    Histogram hist_temp_;       // histogramm of the template image
    cv::Mat weight_geometirc_;  // geometirc weight, which is a gaussian kernel

    BoundingBox bbox_;  // bounding box of the object to tracking
};