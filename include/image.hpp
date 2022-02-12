#ifndef image_hpp
#define image_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace std;

// using static to avoid duplicated symbol MODE from multiple accessible files
static enum mode {
    BASELINE = 1,
} MODE;

namespace image {

// main process steps
void loadImages(vector<cv::Mat> &images, const char *dirname);
vector<pair<cv::Mat, float>> calculateDistances(cv::Mat &source, vector<cv::Mat> &images, mode MODE);
vector<cv::Mat> sortByDistances(vector<pair<cv::Mat, float>> &imgDists);
void displayResults(vector<cv::Mat> &images);

// specific mode
float baselineMatch(cv::Mat &src, cv::Mat &target);

}  // namespace image

#endif /* image_hpp */