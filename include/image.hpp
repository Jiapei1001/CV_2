#ifndef image_hpp
#define image_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace std;

// using static to avoid duplicated symbol MODE from multiple accessible files
static enum mode {
    BASELINE = 1,
    HISTOGRAM = 2,
} MODE;

namespace image {

// main process steps
vector<pair<cv::Mat, float>> calculateDistances(cv::Mat &source, vector<cv::Mat> &images, mode MODE);
vector<cv::Mat> sortByDistances(vector<pair<cv::Mat, float>> &imgDists);

// specific mode
float baselineMatch(cv::Mat &src, cv::Mat &target);
float histogramMatch(cv::Mat &src, cv::Mat &target);

}  // namespace image

#endif /* image_hpp */