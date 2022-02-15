#include "process.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

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

void process::printModeDescriptions() {
    std::cout << "Keys for modes:" << std::endl;
    std::cout << "a)Baseline comparison \t\t\t\t\t -key '0' :" << std::endl;
    std::cout << "b)3D RGB Histogram comparison \t\t\t\t -key '1' :" << std::endl;
    std::cout << "c)Multiple part comparison \t\t\t\t -key '2' :" << std::endl;
    std::cout << "d)Sobel Texture + 3D RGB Histogram comparison  \t\t -key '3' :" << std::endl;
    std::cout << "e)Custom - Centeral Area + Sobel Texture + Hue Saturation-key '4' :" << std::endl;
    std::cout << "f)2D RG Histogram comparison \t\t\t\t -key '5' :" << std::endl;
    std::cout << "g)Sobel Texture + 2D RG Histogram \t\t\t -key '6' :" << std::endl;
    std::cout << "h)Major Shape Contour comparison \t\t\t -key '7' :" << std::endl;
    std::cout << "i)Gradient Orientation Texture + Hue Saturation \t -key '8' :" << std::endl;
}

// load images from a directory
void process::loadImages(vector<Mat> &images, const char *dirname) {
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;

    printf("Processing directory %s\n\n", dirname);

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
            // printf("processing image file: %s\n", dp->d_name);

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

// display results in separate windows
void process::displayResults(vector<cv::Mat> &results) {
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

// display results in one window
void process::displayResultsInOneWindow(vector<cv::Mat> &results) {
    int numR, numC;

    int resSq = (int)sqrt(results.size());

    // get one picture's dimension
    int singleW = results[0].cols;
    int singleH = results[0].rows;

    numC = resSq;
    // account for extra row(s) for remaining image(s), and round up
    numR = ceil((float)results.size() / (float)numC);

    Mat dstMat(Size(numC * singleW, numR * singleH), CV_8UC3, Scalar(120, 120, 120));

    // assign result to mat
    int currIdx = 0;
    for (int i = 0; i < numR; i++) {
        for (int j = 0; j < numC; j++) {
            if (currIdx == results.size()) {
                break;
            }
            results[currIdx].copyTo(dstMat(Rect(j * singleW, i * singleH, singleW, singleH)));
            currIdx++;
        }
    }

    // scale
    float targetWidth = 1200;
    float scale, targetHeight;
    scale = targetWidth / dstMat.cols;
    targetHeight = dstMat.rows * scale;

    resize(dstMat, dstMat, Size(targetWidth, targetHeight));

    string window_name = "top matched results";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, dstMat);
}