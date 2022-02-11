#ifndef image_hpp
#define image_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace std;

namespace image {

void loadImages(vector<cv::Mat> &images, const char *dirname);
float baselineMatch(cv::Mat &src, cv::Mat &target);

}  // namespace image

#endif /* image_hpp */