#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

int main()
{
    std::string filename = "Cameraman.png";
    cv::Mat img_rgb;
    img_rgb = cv::imread(filename);
    if (img_rgb.empty())
    {
        std::cout << "Can not open the file." << std::endl;
    }

    //Create gray-scale image for computation
    cv::Mat img_gray;
    cv::cvtColor(img_rgb, img_gray, cv::COLOR_RGB2GRAY);

    //Create result matrix
    cv::Mat result, result_norm, result_norm_uint8; //32-bit Float  0-1  0-255
    result = cv::Mat::zeros(img_gray.size(), CV_32FC1);

    //Define harris detector
    int block_size = 2; //size of neighbor window 2*block_size + 1
    int aperture_size = 1; //size of gradient operator(sobel) window
    double k = 0.04; //harris responding coefficient 

    cv::cornerHarris(img_gray, result, block_size, aperture_size, k, cv::BORDER_DEFAULT);

    //Normalizing image to 0-255
    cv::normalize(result, result_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    //Convert float to uint8
    cv::convertScaleAbs(result_norm, result_norm_uint8);

    //Drawing circless around corners
    bool b_mark_corners = true;
    if (b_mark_corners)
    {
        double thresh_harris_res = 200;
        int radius = 10;
        for (int j = 0; j < result_norm.rows; j++)
        {
            for (int i = 0; i < result_norm.cols; i++)
            {
                if ((int)result_norm.at<float>(j, i) > thresh_harris_res)
                {
                    cv::circle(result_norm_uint8, cv::Point(i, j), radius, cv::Scalar(255), 1, 8, 0);
                    cv::circle(img_rgb, cv::Point(i, j), radius, cv::Scalar(0, 255, 255), 1, 4, 0);

                    cv::line(img_rgb, cv::Point(i - radius - 2, j), cv::Point(i + radius + 2, j), cv::Scalar(0, 255, 255), 1, 8, 0);
                    cv::line(img_rgb, cv::Point(i, j - radius - 2), cv::Point(i, j + radius + 2), cv::Scalar(0, 255, 255), 1, 8, 0);
                }
            }
        }
    }

    //Show the result
    cv::imshow("source_img", img_rgb);
    cv::imshow("result_scaled", result_norm_uint8);
    cv::waitKey(0);

    return 0;
}

