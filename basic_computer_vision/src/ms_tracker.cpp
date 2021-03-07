#include "..\include\ms_tracker.h"
#include <iostream>

void MeanShiftTracker::process(const std::vector<cv::Mat>& video) {
    for (int i = 1; i < video.size(); i++) {
        std::cout << "Mean shift tracking frame current id : " << i << '\n';
        cv::Point2f pose_predicted = motion_predictor.next_pos();

        int num_iteration = 30;
        ms_tracking.run(num_iteration, pose_predicted, video[i]);

        cv::Point2f pose_tracked = ms_tracking.get_tracked_object_center();
        motion_predictor.set_observation(pose_tracked);

        ms_tracking.visualize(video[i]);
    }
}
