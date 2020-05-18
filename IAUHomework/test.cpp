#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/xfeatures2d.hpp>

int ssiftMatch(std::string path_left, std::string path_right)
{
    cv::Mat image_left, image_right;
    image_left = cv::imread(path_left, cv::IMREAD_COLOR);
    return 0;
}

int amain()
{
    std::string filename = "Cameraman.png";
    //harrisDetector(filename);
    //siftDetector(filename);
    std::string path_left_image = "../image/tail.bmp";
    //std::string path_right_image = "../image/tail2.bmp";
    std::string path_right_image = "../image/tail3.bmp";
    //std::string path_right_image = "../image/tail4.bmp";

    ssiftMatch(path_left_image, path_right_image);

    return 0;
}
