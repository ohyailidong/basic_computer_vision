#pragma once

#include <numeric>
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