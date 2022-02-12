#include "image.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace image;
using namespace std;

// load images from a directory
void image::loadImages(vector<Mat> &images, const char *dirname) {
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;

    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {
            printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // image path
            // printf("full path name: %s\n", buffer);

            cv::Mat newImage;
            newImage = cv::imread(buffer);

            // check if new Mat is built
            if (newImage.data == NULL) {
                cout << "This new image" << buffer << "cannot be loaded into cv::Mat\n";
                exit(-1);
            }

            // image's data
            // cout << "M = " << endl
            //      << " " << newImage.rowRange(0, 6) << endl
            //      << endl;

            images.push_back(newImage);
        }
    }
}

// main entry - calculate distances from source to targets using input feature mode
vector<pair<cv::Mat, float>> image::calculateDistances(cv::Mat &src, vector<cv::Mat> &images, mode MODE) {
    vector<pair<cv::Mat, float>> imgDists;

    for (int i = 0; i < images.size(); i++) {
        float dist;
        switch (MODE) {
        case BASELINE:
            dist = image::baselineMatch(src, images[i]);
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

// display
void image::displayResults(vector<cv::Mat> &results) {
    float targetWidth = 600;
    float scale, targetHeight;

    for (int i = 0; i < results.size(); i++) {
        scale = targetWidth / results[i].cols;
        targetHeight = results[i].rows * scale;
        cv::resize(results[i], results[i], Size(targetWidth, targetHeight));

        string name = "top " + to_string(i);
        namedWindow(name, WINDOW_AUTOSIZE);
        cv::imshow(name, results[i]);
    }
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
