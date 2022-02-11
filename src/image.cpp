#include "image.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace image;
using namespace std;

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

            printf("full path name: %s\n", buffer);

            cv::Mat newImage;
            newImage = cv::imread(buffer);

            // check if new Mat is built
            if (newImage.data == NULL) {
                cout << "This new image" << buffer << "cannot be loaded into cv::Mat\n";
                exit(-1);
            }
            cout << "M = " << endl
                 << " " << newImage.rowRange(0, 6) << endl
                 << endl;
            images.push_back(newImage);
        }
    }
}

float image::baselineMatch(cv::Mat &src, cv::Mat &target) {
    return 0.0;
}
