#include "image.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace image;
using namespace std;

const int Hsize = 16;
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
            dist = image::compareRGBHist(src, images[i]);
            break;
        case MULTI_HISTOGRAM:
            dist = image::compareMultiRGBHist(src, images[i]);
            break;
        case TEXTURE_COLOR:
            dist = image::compareSobelAndColor(src, images[i]);
            break;
        }
        imgDists.push_back(make_pair(images[i], dist));
    }

    return imgDists;
}

// comparator
bool comparatorByDistance(const pair<cv::Mat, float> &p1, const pair<cv::Mat, float> &p2) {
    return p1.second < p2.second;
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

    return dist;
}

// initialize 2D histogram
float **initialize2dHistogram() {
    // 2-D int array
    float **hist_2d = new float *[Hsize];  // allocates an array of int pointers, one per row

    hist_2d[0] = new float[Hsize * Hsize];  // allocates actual data

    // initialize the row pointers
    for (int i = 1; i < Hsize; i++) {
        // note here hist_2d[0] is going into the allocated data above
        hist_2d[i] = &(hist_2d[0][i * Hsize]);
    }

    // initialize the data to all zeros
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            hist_2d[i][j] = 0.0;
        }
    }
    // Alternative initialize option
    // for (int i = 0; i < Hsize * Hsize; i++) {
    //     hist_2d[0][i] = 0;
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
float image::compareRGBHist(cv::Mat &src, cv::Mat &target) {
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

    // get sum of bins
    float sum_src = 0.0;
    float sum_tar = 0.0;
    for (int i = 0; i < Hsize; i++) {
        for (int j = 0; j < Hsize; j++) {
            for (int k = 0; k < Hsize; k++) {
                sum_src += hist_3d_src[i * Hsize * Hsize + j * Hsize + k];
                sum_tar += hist_3d_tar[i * Hsize * Hsize + j * Hsize + k];
            }
        }
    }

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
                dist += (a - b) * (a - b);
            }
        }
    }

    delete hist_3d_src;
    delete hist_3d_tar;

    return dist;
}

// Match image using multiple parts' histograms - top & bottom
float image::compareMultiRGBHist(cv::Mat &src, cv::Mat &target) {
    // Rect(x, y, width, height). In OpenCV, the data are organized with the first pixel being in the upper left corner.
    // top (0, 0, cols, rows / 2)
    // bottom (0, rows / 2, cols, rows / 2)
    // left (0, 0, cols / 2, rows)
    // right (cols/2, 0, cols / 2, rows)

    cv::Mat srcTop(src, Rect(0, 0, src.cols, src.rows / 2));
    cv::Mat srcBot(src, Rect(0, src.rows / 2, src.cols, src.rows / 2));

    cv::Mat tarTop(target, Rect(0, 0, target.cols, target.rows / 2));
    cv::Mat tarBot(target, Rect(0, target.rows / 2, target.cols, target.rows / 2));

    float dist1 = image::compareRGBHist(srcTop, tarTop);
    float dist2 = image::compareRGBHist(srcBot, tarBot);

    // weight, emphasize the top part
    return dist1 * 8 + dist2 * 2;
}

// Get the average difference between sobelX and sobelY differences
float compareSobelHist(cv::Mat &src, cv::Mat &target) {
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
    cv::Sobel(src_gray, src_grad_x, ddepth, 1, 0, 3);
    cv::Sobel(src_gray, src_grad_y, ddepth, 0, 1, 3);
    cv::Sobel(tar_gray, tar_grad_x, ddepth, 1, 0, 3);
    cv::Sobel(tar_gray, tar_grad_y, ddepth, 0, 1, 3);

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
    int method = HISTCMP_CORREL;
    float x_dist = cv::compareHist(src_hist_x, tar_hist_x, method);
    float y_dist = cv::compareHist(src_hist_y, tar_hist_y, method);

    // Step 4: average
    float dist = (x_dist + y_dist) / 2.0;

    return dist;
}

// Compare Sobel Texture Histogram + Color Histogram
float image::compareSobelAndColor(cv::Mat &src, cv::Mat &target) {
    float colorDist = compareRGBHist(src, target);
    float textureDist = compareSobelHist(src, target);

    return colorDist + textureDist * 4;
}

// Backup - A 2 dimensional histogram G & R match
float histogramMatch2DGR(cv::Mat &src, cv::Mat &target) {
    float dist = 0.0;

    // G, B, R -> G & R 2 dimensional histogram
    float **hist_2d_src = initialize2dHistogram();
    float **hist_2d_tar = initialize2dHistogram();

    // int index = R * Hsize / 256;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // g, b, r
            cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            int gIdx = p[0] * Hsize / 256;
            int rIdx = p[2] * Hsize / 256;
            hist_2d_src[gIdx][rIdx] += 1.0;
        }
    }
    for (int y = 0; y < target.rows; y++) {
        for (int x = 0; x < target.cols; x++) {
            // g, b, r
            cv::Vec3b p = target.at<cv::Vec3b>(y, x);
            int gIdx = p[0] * Hsize / 256;
            int rIdx = p[2] * Hsize / 256;
            hist_2d_tar[gIdx][rIdx] += 1.0;
        }
    }

    // for (int i = 0; i < Hsize; i++) {
    //     for (int j = 0; j < Hsize; j++) {
    //         printf("%.1f ", hist_2d_src[i][j]);
    //     }
    //     printf("\n");
    // }

    int sum_src = src.rows * src.cols;
    int sum_tar = target.rows * target.cols;

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
            dist += (a - b) * (a - b);
        }
    }

    delete hist_2d_src[0];
    delete hist_2d_src;
    delete hist_2d_tar[0];
    delete hist_2d_tar;

    return dist;
}

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