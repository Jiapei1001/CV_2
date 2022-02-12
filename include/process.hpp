#ifndef process_hpp
#define process_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace std;

namespace process {
void loadImages(vector<cv::Mat> &images, const char *dirname);
void displayResults(vector<cv::Mat> &images);

}  // namespace process

#endif /* process_hpp */