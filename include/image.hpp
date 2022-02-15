#ifndef image_hpp
#define image_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace std;

// using static to avoid duplicated symbol MODE from multiple accessible files
static enum mode {
    BASELINE = 0,
    HISTOGRAM = 1,
    MULTI_HISTOGRAM = 2,
    SOBEL_COLOR_RGB = 3,
    CUSTOM = 4,
    RG_HISTOGRAM = 5,
    SOBEL_CHROMA_RG = 6,
    SHAPE = 7,
    GRADIENT_COLOR_HS = 8,
} MODE;

namespace image {

// main process steps
vector<pair<cv::Mat, float>> calculateDistances(cv::Mat &source, vector<cv::Mat> &images, mode MODE);
vector<cv::Mat> sortByDistances(vector<pair<cv::Mat, float>> &imgDists);

// specific mode
float baselineMatch(cv::Mat &src, cv::Mat &target);
float compareRGB(cv::Mat &src, cv::Mat &target);
float compareMultiRGB(cv::Mat &src, cv::Mat &target);
float compareSobel(cv::Mat &src, cv::Mat &target, int ksize);
float compareSobelAndRGB(cv::Mat &src, cv::Mat &target);
float compareCustom(cv::Mat &src, cv::Mat &target);

// RG Chrome
float compare2dChromaRG(cv::Mat &src, cv::Mat &target);
float compareSobelAnd2dChromaRG(cv::Mat &src, cv::Mat &target);

// HS
float compareHsHist(cv::Mat &src, cv::Mat &target, float hrange[], float srange[]);
float compareSobelAndHS(cv::Mat &src, cv::Mat &target, float hrange[], float srange[]);

// Gradient
float compareGradient(cv::Mat &src, cv::Mat &target, int ksize);
float compareGradientAndHS(cv::Mat &src, cv::Mat &target);

// Match Shape
float compareShape(cv::Mat &src, cv::Mat &target);

}  // namespace image

#endif /* image_hpp */