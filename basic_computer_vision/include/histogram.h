#pragma once
#include <cmath>
#include <vector>

class Histogram {
   public:
    Histogram(int num_bin, double min_value, double max_value)
        : width_bin_((max_value - min_value + 1) / num_bin), hist_(num_bin) {
    }
 
    void add_data(double value, double weight = 1) {
        int bin = get_bin_id(value);
        hist_[bin] += weight;
    }
    int num_bin() const {
        return hist_.size();
    }
    int get_bin_id(double value) const {
        return std::floor(value / width_bin_);
    }
    double get_bin_height(double id_bin) const {
        return hist_[id_bin];
    }
    std::vector<double> get_hist() const {
        return hist_;
    }

   private:
    double width_bin_;  // the width of a bin
    std::vector<double> hist_;
};