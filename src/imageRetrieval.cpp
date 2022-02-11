/*
  Identify image fils in a directory
*/
#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>

#include "image.hpp"

using namespace cv;
using namespace std;
using namespace image;

enum mode {
    BASELINE = 1,
} MODE;

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
    char dirname[256];
    cv::Mat source;
    vector<cv::Mat> images;
    int feature;

    // check for sufficient arguments
    if (argc < 4) {
        cout << "usage: %s <source> <directory path> <mode>\n";
        exit(-1);
    }

    // get the source image
    source = cv::imread(argv[1]);
    printf("Processing source %s\n", argv[1]);
    if (source.data == NULL) {
        printf("Unable to read query image %s\n", argv[1]);
        exit(-1);
    }

    // get the directory path & load images
    strcpy(dirname, argv[2]);
    image::loadImages(images, dirname);
    cout << "image numbers" << images.size() << "\n";

    // choose feature mode
    feature = atoi(argv[3]);
    switch (feature) {
    case 0:
        MODE = BASELINE;
        break;
    default:
        MODE = BASELINE;
        break;
    }

    // // process mode
    // switch (MODE) {
    // case BASELINE:
    //     /* code */
    //     break;
    // }

    printf("Terminating\n");

    return (0);
}
