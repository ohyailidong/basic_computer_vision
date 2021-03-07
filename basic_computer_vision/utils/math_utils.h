#pragma once

#include <array>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <random>
#include <vector>

template <typename T>
T mean(const std::vector<T>& vec) {
    assert(!vec.empty());
    T sum = std::accumulate(vec.begin(), vec.end(), vec[0]) - vec[0];

    return sum / vec.size();
}

template <>
cv::Vec2f mean(const std::vector<cv::Vec2f>& vec) {
    assert(!vec.empty());
    cv::Vec2f sum = std::accumulate(vec.begin(), vec.end(), vec[0]) - vec[0];

    return cv::Vec2f(sum[0] / vec.size(), sum[1] / vec.size());
}

std::vector<int> generate_random_data(int num, int min, int max) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int> uniform_dist(min, max);
    std::vector<int> data(num);
    for (int i = 0; i < num; i++) {
        data[i] = uniform_dist(gen);
    }
    return data;
}

std::vector<float> generate_random_data(int num, float min, float max) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> uniform_dist(min, max);
    std::vector<float> data(num);
    for (int i = 0; i < num; i++) {
        data[i] = uniform_dist(gen);
    }
    return data;
}

float generate_random_data(float min, float max) {
    return generate_random_data(1, min, max)[0];
}

int generate_random_data(int min, int max) {
    return generate_random_data(1, min, max)[0];
}

template <typename T, int Dim>
std::vector<std::array<T, Dim>> generate_gauss_data(int num, const std::array<T, Dim>& means,
                                                    const std::array<T, Dim>& stddev) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::array<std::normal_distribution<T>, Dim> normal_dists;
    for (int i = 0; i < Dim; i++) {
        normal_dists[i] = std::normal_distribution<T>(means[i], stddev[i]);
    }

    std::vector<std::array<T, Dim>> result(num);
    for (int n = 0; n < num; n++) {
        std::array<T, Dim> data;
        for (int i = 0; i < Dim; i++) {
            data[i] = normal_dists[i](gen);
        }
        result[n] = (data);
    }

    return result;
}

template <int Dim>
std::vector<std::array<int, Dim>> generate_nd_data(int num, int min, int max) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::array<std::uniform_int_distribution<int>, Dim> uniform_dists;
    for (int i = 0; i < Dim; i++) {
        uniform_dists[i] = std::uniform_int_distribution<int>(min, max);
    }

    std::vector<std::array<int, Dim>> result(num);
    for (int n = 0; n < num; n++) {
        std::array<int, Dim> data;
        for (int i = 0; i < Dim; i++) {
            data[i] = uniform_dists[i](gen);
        }
        result[n] = (data);
    }

    return result;
}

std::vector<double> generate_gmm_data(int num_want, const std::vector<double>& means, const std::vector<double>& vars,
                                      const std::vector<double>& weights) {
    assert(means.size() == vars.size() && means.size() == weights.size());
    int sz = means.size();
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::vector<std::normal_distribution<double>> dists;
    for (int i = 0; i < sz; i++) {
        dists.emplace_back(means[i], vars[i]);
    }

    std::vector<double> result;

    for (int n = 0; n < num_want; n++) {
        double tmp = 0.0;
        for (int i = 0; i < sz; i++) {
            tmp += weights[i] * dists[i](gen);
        }
        result.push_back(tmp);
    }

    return result;
}