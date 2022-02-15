#include "image.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace image;
using namespace std;

const int Hsize = 10;
float range[] = {0, 256};
const float *HistRange = {range};

// main entry - calculate distances from source to targets using input feature mode
vector<pair<cv::Mat, float>> image::calculateDistances(cv::Mat &src, vector<cv::Mat> &images, mode MODE) {
    vector<pair<cv::Mat, float>> imgDists;

    for (int i = 0; i < images.size(); i++) {
        float dist;
        switch (MODE) {
        case BASELINE:
            dist = image::baselineMatch(src, images[i]);
            break;
        case HISTOGRAM:
            dist = image::compareRGB(src, images[i]);
            break;
        case MULTI_HISTOGRAM:
            dist = image::compareMultiRGB(src, images[i]);
            break;
        case SOBEL_COLOR_RGB:
            dist = image::compareSobelAndRGB(src, images[i]);
            break;
        case CUSTOM:
            dist = image::compareCustom(src, images[i]);
            break;
        case RG_HISTOGRAM:
            dist = image::compare2dChromaRG(src, images[i]);
            break;
        case SOBEL_CHROMA_RG:
            dist = image::compareSobelAnd2dChromaRG(src, images[i]);
            break;
        case SHAPE:
            dist = image::compareShape(src, images[i]);
            break;
        case GRADIENT_COLOR_HS:
            dist = image::compareGradientAndHS(src, images[i]);
            break;
        }

        imgDists.push_back(make_pair(images[i], dist));
    }

    return imgDists;
}

// Comparator
// OpenCV doc: https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html
// Intersection(method = CV_COMP_INTERSECT), d(H1, H2) =∑Imin(H1(I), H2(I))
// Thus, the bigger the value is, the matching is closer
bool comparatorByDistance(const pair<cv::Mat, float> &p1, const pair<cv::Mat, float> &p2) {
    // intersection comparison
    return p1.second > p2.second;
}

// sort image list by comparator
vector<cv::Mat> image::sortByDistances(vector<pair<cv::Mat, float>> &imgDists) {
    sort(imgDists.begin(), imgDists.end(), comparatorByDistance);

    vector<cv::Mat> sorted;
    for (int i = 0; i < imgDists.size(); i++) {
        sorted.push_back(imgDists[i].first);
    }

    return sorted;
}

// 9 x 9 grid
float image::baselineMatch(cv::Mat &src, cv::Mat &target) {
    int sy = src.rows / 2 - 4;
    int sx = src.cols / 2 - 4;
    int ty = target.rows / 2 - 4;
    int tx = target.cols / 2 - 4;

    float dist = 0.0;
    // 9 x 9
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            cv::Vec3b s = src.at<cv::Vec3b>(sy + i, sx + j);
            cv::Vec3b t = target.at<cv::Vec3b>(ty + i, tx + j);

            dist += (s[0] - t[0]) * (s[0] - t[0]);
            dist += (s[1] - t[1]) * (s[1] - t[1]);
            dist += (s[2] - t[2]) * (s[2] - t[2]);
        }
    }

    // intersection comparison, thus negative
    return -dist;
}

// initialize 2D histogram
float **initialize2dHistogram() {
    // 2-D int array
    float **hist_2d = new float *[Hsize];  // allocates an array of int pointers, one per row

    // hist_2d[0] = new float[Hsize * Hsize];  // allocates actual data

    // initialize the row pointers
    for (int i = 0; i < Hsize; i++) {
        // note here hist_2d[0] is going into the allocated data above
        // hist_2d[i] = &(hist_2d[0][i * Hsize]);
        hist_2d[i] = new float[Hsize];
    }

    // initialize the data to all zeros
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            hist_2d[i][j] = 0.0;
        }
    }
    // Alternative initialize option
    // for (int i = 0; i < Hsize * Hsize; i++) {
    //     hist_2d[0][i] = 0.0;
    // }

    return hist_2d;
}

// initialize 3D histogram
float *initialize3dHistogram() {
    // 3-D int array
    float *hist_3d = new float[Hsize * Hsize * Hsize];  // allocates an array of int pointers, one per row

    // initialize the data to all zeros
    for (int i = 0; i < Hsize * Hsize * Hsize; i++) {
        hist_3d[i] = 0.0;
    }

    return hist_3d;
}

// A 3 dimensional R G B histogram match
float image::compareRGB(cv::Mat &src, cv::Mat &target) {
    float dist = 0.0;

    // G, B, R -> 3 dimensional histogram
    float *hist_3d_src = initialize3dHistogram();
    float *hist_3d_tar = initialize3dHistogram();

    // int index = (R * Hsize) / 256;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // g, b, r
            cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            int gIdx = (p[0] * Hsize) / 256;
            int bIdx = (p[1] * Hsize) / 256;
            int rIdx = (p[2] * Hsize) / 256;
            hist_3d_src[gIdx * Hsize * Hsize + bIdx * Hsize + rIdx] += 1.0;
        }
    }
    for (int y = 0; y < target.rows; y++) {
        for (int x = 0; x < target.cols; x++) {
            // g, b, r
            cv::Vec3b p = target.at<cv::Vec3b>(y, x);
            int gIdx = (p[0] * Hsize) / 256;
            int bIdx = (p[1] * Hsize) / 256;
            int rIdx = (p[2] * Hsize) / 256;
            hist_3d_tar[gIdx * Hsize * Hsize + bIdx * Hsize + rIdx] += 1.0;
        }
    }

    // get sum of bins, one pixel contribute one
    float sum_src = src.rows * src.cols;
    float sum_tar = target.rows * target.cols;

    // normalize
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            for (int k = 0; k < Hsize; k++) {
                hist_3d_src[i * Hsize * Hsize + j * Hsize + k] /= sum_src;
                hist_3d_tar[i * Hsize * Hsize + j * Hsize + k] /= sum_tar;
            }
        }
    }

    // compare
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            for (int k = 0; k < Hsize; k++) {
                float a = hist_3d_src[i * Hsize * Hsize + j * Hsize + k];
                float b = hist_3d_tar[i * Hsize * Hsize + j * Hsize + k];
                // Intersection comparison, match the method used when comparing two Sobel histograms in OpenCV
                dist += std::min(a, b);
            }
        }
    }

    delete hist_3d_src;
    delete hist_3d_tar;

    return dist;
}

// Match image using multiple parts' histograms - top & bottom
float image::compareMultiRGB(cv::Mat &src, cv::Mat &target) {
    // Rect(x, y, width, height). In OpenCV, the data are organized with the first pixel being in the upper left corner.
    // top (0, 0, cols, rows / 2)
    // bottom (0, rows / 2, cols, rows / 2)
    // left (0, 0, cols / 2, rows)
    // right (cols/2, 0, cols / 2, rows)

    cv::Mat srcTop(src, Rect(0, 0, src.cols, src.rows / 2));
    cv::Mat srcBot(src, Rect(0, src.rows / 2, src.cols, src.rows / 2));

    cv::Mat tarTop(target, Rect(0, 0, target.cols, target.rows / 2));
    cv::Mat tarBot(target, Rect(0, target.rows / 2, target.cols, target.rows / 2));

    float dist1 = image::compareRGB(srcTop, tarTop);
    float dist2 = image::compareRGB(srcBot, tarBot);

    srcTop.release();
    srcBot.release();
    tarTop.release();
    tarBot.release();

    // weight, emphasize the top part
    return dist1 * 2 + dist2;
}

// Get the average difference between sobelX and sobelY differences
float image::compareSobel(cv::Mat &src, cv::Mat &target, int ksize) {
    // get sobelX, sobelY
    // compare sobelX between src and target
    // compare sobelY between src and target
    // average of the 2 results

    // Step One: get sobelX & sobelY - refer to OpenCV doc

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::Mat src_copy, tar_copy;
    cv::GaussianBlur(src, src_copy, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cv::GaussianBlur(target, tar_copy, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Convert the image to grayscale
    cv::Mat src_gray, tar_gray;
    cv::cvtColor(src_copy, src_gray, COLOR_BGR2GRAY);
    cv::cvtColor(tar_copy, tar_gray, COLOR_BGR2GRAY);

    // Sobel
    cv::Mat src_grad_x, src_grad_y, tar_grad_x, tar_grad_y;
    int ddepth = CV_16S;
    cv::Sobel(src_gray, src_grad_x, ddepth, 1, 0, ksize);
    cv::Sobel(src_gray, src_grad_y, ddepth, 0, 1, ksize);
    cv::Sobel(tar_gray, tar_grad_x, ddepth, 1, 0, ksize);
    cv::Sobel(tar_gray, tar_grad_y, ddepth, 0, 1, ksize);

    // Converting back to CV_8U
    cv::Mat src_x_abs, src_y_abs, tar_x_abs, tar_y_abs;
    cv::convertScaleAbs(src_grad_x, src_x_abs);
    cv::convertScaleAbs(src_grad_y, src_y_abs);
    cv::convertScaleAbs(tar_grad_x, tar_x_abs);
    cv::convertScaleAbs(tar_grad_y, tar_y_abs);

    // Calculate the weighted sum
    cv::Mat src_grad, tar_grad;
    cv::addWeighted(src_x_abs, 0.5, src_y_abs, 0.5, 0, src_grad);
    cv::addWeighted(tar_x_abs, 0.5, tar_y_abs, 0.5, 0, tar_grad);

    // Step 2: get Sobel histogram
    // as gray, only one channel, cannot use the RGB histogram comparison
    cv::Mat src_hist, tar_hist;
    // source array; 1 as number of source array; 0 as channel dimension, here gray as single channel;
    // Mat() a mask onto source array, here not used;
    // result mat; 1 as histogram dimension, here as 1;
    // Hsize as # of bins; HistRange as {0, 256}
    cv::calcHist(&src_grad, 1, 0, Mat(), src_hist, 1, &Hsize, &HistRange);
    cv::calcHist(&tar_grad, 1, 0, Mat(), tar_hist, 1, &Hsize, &HistRange);

    // normalize, one pixel contribute to one, sum as total # of pixels
    // float src_sum = (float)src_hist.rows * src_hist.cols;
    // float tar_sum = (float)tar_hist.rows * tar_hist.cols;
    // src_hist /= src_sum;
    // tar_hist /= tar_sum;
    // cv::normalize(src_hist, src_hist, 0, src.rows * src.cols, NORM_MINMAX, -1, Mat());
    // cv::normalize(tar_hist, tar_hist, 0, target.rows * target.cols, NORM_MINMAX, -1, Mat());

    cv::normalize(src_hist, src_hist, 0.0, 1.0, NORM_MINMAX, -1, Mat());
    cv::normalize(tar_hist, tar_hist, 0.0, 1.0, NORM_MINMAX, -1, Mat());

    // Step 3: compare Sobel histogram
    int method = HISTCMP_INTERSECT;
    float dist = (float)cv::compareHist(src_hist, tar_hist, method);

    src_copy.release();
    tar_copy.release();
    src_gray.release();
    tar_gray.release();

    src_grad_x.release();
    src_grad_y.release();
    tar_grad_x.release();
    tar_grad_y.release();

    src_x_abs.release();
    src_y_abs.release();
    tar_x_abs.release();
    tar_y_abs.release();

    src_grad.release();
    tar_grad.release();

    src_hist.release();
    tar_hist.release();

    return dist;
}

// Compare Sobel Texture Histogram + Color 3D RGB Histogram
float image::compareSobelAndRGB(cv::Mat &src, cv::Mat &target) {
    float colorDist = compareRGB(src, target);
    float textureDist3 = compareSobel(src, target, 3);
    float textureDist5 = compareSobel(src, target, 5);
    // float textureDist7 = compareSobel(src, target, 7);

    // more weight on texture
    // return colorDist + textureDist;
    return colorDist + textureDist3 + textureDist5;
}

// Custom - Divide the mat to a 4 * 8 grid + Sobel Texture + Hue & Saturation
// focus on yellow hue range
float image::compareCustom(cv::Mat &src, cv::Mat &target) {
    // hue - focus on yellow hue
    float hrange[] = {22, 38};
    // saturation
    float srange[] = {0, 256};

    // compare whole mats by Sobel Texture + Hue & Saturation
    float dist_whole = image::compareSobelAndHS(src, target, hrange, srange);
    // float dist_whole = image::compareSobelAndRGB(src, target);

    // divide the mat into 4 rows * 8 cols grid
    // get the centered frame with 4 grid positions as (1, 2), (1, 6), (3, 2), (3, 6)

    // Rect(x, y, width, height). In OpenCV, the data are organized with the first pixel being in the upper left corner.
    // top (0, 0, cols, rows / 2)
    // bottom (0, rows / 2, cols, rows / 2)
    // left (0, 0, cols / 2, rows)
    // right (cols/2, 0, cols / 2, rows)
    int src_grid_width = src.cols / 8;
    int src_grid_height = src.rows / 4;
    cv::Mat srcCenter(src, Rect(src_grid_width * 2, src_grid_height, src_grid_width * 4, src_grid_height * 2));

    int tar_grid_width = target.cols / 8;
    int tar_grid_height = target.rows / 4;
    cv::Mat tarCenter(target, Rect(tar_grid_width * 2, tar_grid_height, tar_grid_width * 4, tar_grid_height * 2));

    // compare center mats by Sobel Texture + Hue & Saturation
    float dist_center = image::compareSobelAndHS(srcCenter, tarCenter, hrange, srange);
    // float dist_center = image::compareSobelAndRGB(srcCenter, tarCenter);

    // weight more on dist_center
    return dist_whole + dist_center * 10;
}

// Compare 2D RG Chromacity Histogram
float image::compare2dChromaRG(cv::Mat &src, cv::Mat &target) {
    float dist = 0.0;

    // G, B, R -> G & R 2 dimensional histogram
    // float **hist_2d_src = initialize2dHistogram();
    // float **hist_2d_tar = initialize2dHistogram();

    auto hist_2d_src = new float[Hsize][Hsize]();
    auto hist_2d_tar = new float[Hsize][Hsize]();

    // B G R
    // int index = R * Hsize / 256;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // g, b, r
            cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            // +1 to avoid two special cases: R, G, B = (0, 0, 0), or the case where two out of three values are 0
            int sumrgb = p[0] + p[1] + p[2] + 1;
            // NOTE: HERE HSIZE must be put in front!! cannot be (p[1] * sumrgb) / Hsize!! as this will make every index as 0
            int gIdx = (p[1] * Hsize) / sumrgb;
            int rIdx = (p[2] * Hsize) / sumrgb;
            hist_2d_src[gIdx][rIdx] += 1.0;
        }
    }
    for (int y = 0; y < target.rows; y++) {
        for (int x = 0; x < target.cols; x++) {
            // g, b, r
            cv::Vec3b p = target.at<cv::Vec3b>(y, x);
            int sumrgb = p[0] + p[1] + p[2] + 1;
            // NOTE: HERE HSIZE must be put in front!! cannot be (p[1] * sumrgb) / Hsize!! as this will make every index as 0
            int gIdx = (p[1] * Hsize) / sumrgb;
            int rIdx = (p[2] * Hsize) / sumrgb;
            hist_2d_tar[gIdx][rIdx] += 1.0;
        }
    }

    // sum, # of pixels
    int sum_src = src.rows * src.cols;
    int sum_tar = target.rows * target.cols;
    // float sum_src = 0.0;
    // float sum_tar = 0.0;
    // for (int i = 0; i < Hsize; i++) {
    //     for (int j = 0; j < Hsize; j++) {
    //         sum_src += hist_2d_src[i][j];
    //         sum_tar += hist_2d_tar[i][j];
    //     }
    // }

    // normalize histogram
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            hist_2d_src[i][j] /= sum_src;
            hist_2d_tar[i][j] /= sum_tar;
        }
    }

    // calculate difference
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            float a = hist_2d_src[i][j];
            float b = hist_2d_tar[i][j];
            // intersection comparison
            dist += std::min(a, b);
            // dist += (a - b) * (a - b);
        }
    }

    // delete hist_2d_src[0];
    // delete hist_2d_src;
    // delete hist_2d_tar[0];
    // delete hist_2d_tar;

    return dist;
}

// Compare Sobel Texture Histogram + Chroma 2D RG Histogram
float image::compareSobelAnd2dChromaRG(cv::Mat &src, cv::Mat &target) {
    float chromaRGDist = image::compare2dChromaRG(src, target);
    float sobelDist = image::compareSobel(src, target, 3);

    // weight more on sobel texture
    return chromaRGDist + sobelDist;
}

// Compare Hue Saturation
float image::compareHsHist(cv::Mat &src, cv::Mat &target, float hrange[], float srange[]) {
    cv::Mat srcHsv;
    cv::Mat tarHsv;
    // convert to hsv
    cv::cvtColor(src, srcHsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(target, tarHsv, cv::COLOR_BGR2HSV);

    int histSize[] = {Hsize, Hsize};
    // hue
    // float hrange[] = {34, 46};
    // saturation
    // float srange[] = {0, 256};
    const float *range[] = {hrange, srange};

    cv::Mat srcHist;
    cv::Mat tarHist;
    int channels[] = {0, 1};
    cv::calcHist(&srcHsv, 1, channels, cv::Mat(), srcHist, 2, histSize, range, true, false);
    cv::calcHist(&tarHsv, 1, channels, cv::Mat(), tarHist, 2, histSize, range, true, false);

    // normalize
    // srcHist = srcHist / (src.rows * src.cols);
    // tarHist = tarHist / (target.rows * target.cols);
    normalize(srcHist, srcHist, 0, 1, NORM_MINMAX, -1);
    normalize(tarHist, tarHist, 0, 1, NORM_MINMAX, -1);

    // compare
    double dist = cv::compareHist(srcHist, tarHist, cv::HISTCMP_INTERSECT);

    srcHsv.release();
    tarHsv.release();
    srcHist.release();
    tarHist.release();

    return (float)dist;
}

// Compare Sobel Texture Histogram + Hue Satuation Histogram
float image::compareSobelAndHS(cv::Mat &src, cv::Mat &target, float hrange[], float srange[]) {
    float hsDist = image::compareHsHist(src, target, hrange, srange);
    float sobelDist = image::compareSobel(src, target, 3);

    // weight more on hue
    return hsDist + sobelDist;
}

// comparison function object
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    // bigger one first
    return (i > j);
}

float image::compareShape(cv::Mat &src, cv::Mat &target) {
    cv::Mat src_gray, tar_gray;
    cv::cvtColor(src, src_gray, COLOR_BGR2GRAY);
    cv::cvtColor(target, tar_gray, COLOR_BGR2GRAY);

    cv::Mat src_copy, tar_copy;
    cv::threshold(src_gray, src_copy, 128, 255, THRESH_BINARY);
    cv::threshold(tar_gray, tar_copy, 128, 255, THRESH_BINARY);

    vector<vector<Point>> src_contours;
    vector<Vec4i> src_hierarchy;
    cv::findContours(src_copy, src_contours, src_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> tar_contours;
    vector<Vec4i> tar_hierarchy;
    cv::findContours(tar_copy, tar_contours, tar_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    if (src_contours.size() > 1 && tar_contours.size() > 1) {
        std::sort(src_contours.begin(), src_contours.end(), compareContourAreas);
        std::sort(tar_contours.begin(), tar_contours.end(), compareContourAreas);

        // the closer, the smaller
        double d = cv::matchShapes(src_contours[0], tar_contours[0], 1, 0.0);
        float dist = (float)d;

        // making the closer ones bigger
        return -dist;
    } else {
        return -100.0;
    }
}

// Compare Gradient Orientation
float image::compareGradient(cv::Mat &src, cv::Mat &target, int ksize) {
    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::Mat src_copy, tar_copy;
    cv::GaussianBlur(src, src_copy, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cv::GaussianBlur(target, tar_copy, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Convert the image to grayscale
    cv::Mat src_gray, tar_gray;
    cv::cvtColor(src_copy, src_gray, COLOR_BGR2GRAY);
    cv::cvtColor(tar_copy, tar_gray, COLOR_BGR2GRAY);

    // Sobel
    cv::Mat src_grad_x, src_grad_y, tar_grad_x, tar_grad_y;
    int ddepth = CV_16S;
    cv::Sobel(src_gray, src_grad_x, ddepth, 1, 0, ksize);
    cv::Sobel(src_gray, src_grad_y, ddepth, 0, 1, ksize);
    cv::Sobel(tar_gray, tar_grad_x, ddepth, 1, 0, ksize);
    cv::Sobel(tar_gray, tar_grad_y, ddepth, 0, 1, ksize);

    // Converting back to CV_32f
    src_grad_x.convertTo(src_grad_x, CV_32F);
    src_grad_y.convertTo(src_grad_y, CV_32F);
    tar_grad_x.convertTo(tar_grad_x, CV_32F);
    tar_grad_y.convertTo(tar_grad_y, CV_32F);

    // Angle
    cv::Mat src_angle, tar_angle;
    src_angle.create(src.size(), src.type());
    tar_angle.create(target.size(), target.type());
    cv::phase(src_grad_x, src_grad_y, src_angle, true);
    cv::phase(tar_grad_x, tar_grad_y, tar_angle, true);

    // Step 2: get histogram
    cv::Mat src_hist, tar_hist;
    // source array; 1 as number of source array; 0 as channel dimension, here gray as single channel;
    // Mat() a mask onto source array, here not used;
    // result mat; 1 as histogram dimension, here as 1;
    // Hsize as # of bins; HistRange as {0, 256}
    cv::calcHist(&src_angle, 1, 0, Mat(), src_hist, 1, &Hsize, &HistRange);
    cv::calcHist(&tar_angle, 1, 0, Mat(), tar_hist, 1, &Hsize, &HistRange);

    // normalize
    cv::normalize(src_hist, src_hist, 0.0, 1.0, NORM_MINMAX, -1, Mat());
    cv::normalize(tar_hist, tar_hist, 0.0, 1.0, NORM_MINMAX, -1, Mat());

    // Step 3: compare Sobel histogram
    int method = HISTCMP_INTERSECT;
    float dist = (float)cv::compareHist(src_hist, tar_hist, method);

    return dist;
}

float image::compareGradientAndHS(cv::Mat &src, cv::Mat &target) {
    // hue
    float hrange[] = {0, 180};
    // saturation
    float srange[] = {0, 256};
    float colorDist = image::compareHsHist(src, target, hrange, srange);

    float gradientDist3 = image::compareGradient(src, target, 3);
    float gradientDist5 = image::compareGradient(src, target, 5);
    float gradientDist7 = image::compareGradient(src, target, 7);
    float gradientDist9 = image::compareGradient(src, target, 9);

    return colorDist + gradientDist3 + gradientDist5 + gradientDist7 + gradientDist9;
}

/*********************** BACK UP ***********************************
// Backup - A 3 dimensional histogram match, RGB 3 separate channels
float histogramMatchRGB(cv::Mat &src, cv::Mat &target) {
    float dist = 0.0;

    float **hist_2d_src = new float *[3];  // allocates an array of int pointers, one per row
    float **hist_2d_tar = new float *[3];

    hist_2d_src[0] = new float[3 * Hsize];  // allocates actual data
    hist_2d_tar[0] = new float[3 * Hsize];

    // initialize the row pointers
    for (int i = 1; i < 3; i++) {
        // note here hist_2d[0] is going into the allocated data above
        hist_2d_src[i] = &(hist_2d_src[0][i * Hsize]);
        hist_2d_tar[i] = &(hist_2d_tar[0][i * Hsize]);
    }

    // int index = R * Hsize / 256;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // g, b, r
            cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            int gIdx = p[0] * Hsize / 256;
            int bIdx = p[1] * Hsize / 256;
            int rIdx = p[2] * Hsize / 256;
            hist_2d_src[0][gIdx] += 1.0;
            hist_2d_src[1][bIdx] += 1.0;
            hist_2d_src[2][rIdx] += 1.0;
        }
    }
    for (int y = 0; y < target.rows; y++) {
        for (int x = 0; x < target.cols; x++) {
            // g, b, r
            cv::Vec3b p = target.at<cv::Vec3b>(y, x);
            int gIdx = p[0] * Hsize / 256;
            int bIdx = p[1] * Hsize / 256;
            int rIdx = p[2] * Hsize / 256;
            hist_2d_tar[0][gIdx] += 1.0;
            hist_2d_tar[1][bIdx] += 1.0;
            hist_2d_tar[2][rIdx] += 1.0;
        }
    }

    // test print
    // for (int i = 0; i < Hsize; i++) {
    //     for (int j = 0; j < Hsize; j++) {
    //         printf("%.1f ", hist_2d_src[i][j]);
    //     }
    //     printf("\n");
    // }

    int sum_src = src.rows * src.cols;
    int sum_tar = target.rows * target.cols;

    // normalize histogram
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < Hsize; j++) {
            hist_2d_src[i][j] /= sum_src;
            hist_2d_tar[i][j] /= sum_tar;
        }
    }

    // calculate difference
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            float a = hist_2d_src[i][j];
            float b = hist_2d_tar[i][j];
            dist += (a - b) * (a - b);
        }
    }

    // delete hist_2d_src;
    // delete hist_2d_tar;

    return dist;
}


// Back up - Get the average difference between sobelX and sobelY differences
float image::compareSobelHist(cv::Mat &src, cv::Mat &target, int ksize) {
    // get sobelX, sobelY
    // compare sobelX between src and target
    // compare sobelY between src and target
    // average of the 2 results

    // Step One: get sobelX & sobelY - refer to OpenCV doc

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::Mat src_copy, tar_copy;
    cv::GaussianBlur(src, src_copy, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cv::GaussianBlur(target, tar_copy, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Convert the image to grayscale
    cv::Mat src_gray, tar_gray;
    cv::cvtColor(src_copy, src_gray, COLOR_BGR2GRAY);
    cv::cvtColor(tar_copy, tar_gray, COLOR_BGR2GRAY);

    // Sobel
    cv::Mat src_grad_x, src_grad_y, tar_grad_x, tar_grad_y;
    int ddepth = CV_16S;
    cv::Sobel(src_gray, src_grad_x, ddepth, 1, 0, ksize);
    cv::Sobel(src_gray, src_grad_y, ddepth, 0, 1, ksize);
    cv::Sobel(tar_gray, tar_grad_x, ddepth, 1, 0, ksize);
    cv::Sobel(tar_gray, tar_grad_y, ddepth, 0, 1, ksize);

    // Converting back to CV_8U
    cv::Mat src_x_abs, src_y_abs, tar_x_abs, tar_y_abs;
    cv::convertScaleAbs(src_grad_x, src_x_abs);
    cv::convertScaleAbs(src_grad_y, src_y_abs);
    cv::convertScaleAbs(tar_grad_x, tar_x_abs);
    cv::convertScaleAbs(tar_grad_y, tar_y_abs);

    // Step 2: get Sobel histogram
    // as gray, only one channel, cannot use the RGB histogram comparison
    cv::Mat src_hist_x, src_hist_y, tar_hist_x, tar_hist_y;
    // source array; 1 as number of source array; 0 as channel dimension, here gray as single channel;
    // Mat() a mask onto source array, here not used;
    // result mat; 1 as histogram dimension, here as 1;
    // Hsize as # of bins; HistRange as {0, 256}
    cv::calcHist(&src_x_abs, 1, 0, Mat(), src_hist_x, 1, &Hsize, &HistRange);
    cv::calcHist(&src_y_abs, 1, 0, Mat(), src_hist_y, 1, &Hsize, &HistRange);
    cv::calcHist(&tar_x_abs, 1, 0, Mat(), tar_hist_x, 1, &Hsize, &HistRange);
    cv::calcHist(&tar_y_abs, 1, 0, Mat(), tar_hist_y, 1, &Hsize, &HistRange);

    // normalize, one pixel contribute to one, sum as total # of pixels
    float src_sum = (float)src_hist_x.size().width * src_hist_x.size().height;
    float tar_sum = (float)tar_hist_x.size().width * tar_hist_x.size().height;
    src_hist_x /= src_sum;
    src_hist_y /= src_sum;
    tar_hist_x /= tar_sum;
    tar_hist_y /= tar_sum;

    // Step 3: compare Sobel histogram
    int method = HISTCMP_INTERSECT;
    float x_dist = cv::compareHist(src_hist_x, tar_hist_x, method);
    float y_dist = cv::compareHist(src_hist_y, tar_hist_y, method);

    // Step 4: average
    float dist = (x_dist + y_dist) / 2.0;

    return dist;
}
*/