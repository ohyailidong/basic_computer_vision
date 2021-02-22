#include "..\include\optical_flow_tracker.h"
#include "..\include\bounding_box.h"
#include "..\utils\opencv_utils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

void apply_histo_equalization_around(BoundingBox bbox, cv::Mat img, int ratio_flattenning) {
    cv::Rect2i rect_local(bbox.top_left().x - 0.5 * ratio_flattenning * bbox.width(),
                          bbox.top_left().y - 0.5 * ratio_flattenning * bbox.height(),
                          (1 + ratio_flattenning) * bbox.width(), (1 + ratio_flattenning) * bbox.height());
    cv::Rect2i rect_intersection =
        get_intersection_from_ul(img, rect_local.tl().x, rect_local.tl().y, rect_local.width, rect_local.height);

    cv::Mat img_intersection = get_sub_image_around(img, rect_intersection.tl().x + 0.5 * rect_intersection.width,
                                                    rect_intersection.tl().y + 0.5 * rect_intersection.height,
                                                    rect_intersection.width, rect_intersection.height);

    cv::equalizeHist(img_intersection, img_intersection);
    img_intersection.copyTo(img(rect_intersection));
}

void OpticalFlowTracker::process(BoundingBox initial_bbox, const std::vector<cv::Mat>& videos) {
    assert(!videos.empty());

    cv::Mat last_img = videos[0];

    apply_histo_equalization_around(initial_bbox, videos[0], 0.06);
    feature_points_manager_.initialize(videos[0], initial_bbox);

    for (int i = 1; i < videos.size(); i++) {
        std::vector<cv::Point2f> prev_feature_points = feature_points_manager_.get_feature_points();

        std::vector<cv::Point2f> curr_feature_points;
        std::vector<uchar> status;
        std::vector<float> err;

        BoundingBox bbox = feature_points_manager_.get_bbox();

        apply_histo_equalization_around(bbox, videos[i], 0.1);

        // Optical Flow
		cv::calcOpticalFlowPyrLK(videos[i - 1], videos[i], prev_feature_points, curr_feature_points, status,
			err, cv::Size(21,21), 3);

        feature_points_manager_.process_feature_points(videos[i], curr_feature_points, status);

        last_img = videos[i];
    }
}
