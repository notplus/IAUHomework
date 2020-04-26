#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/xfeatures2d.hpp>

int siftDetector(std::string path)
{
    cv::Mat image_rgb,image;
    image_rgb = cv::imread(path);

    if (image_rgb.empty())
    {
        std::cout << "Fail to read the image: " << path << std::endl;
        return -1;
    }

    cv::cvtColor(image_rgb, image, cv::COLOR_RGB2GRAY);

    //cv::keyPoint --SIFT corner, SURF, FAST, ORB
    std::vector<cv::KeyPoint> keypoints; //(x, y, scale, angel)

    double thresh_contrast = 0.04; //小于此阈值的特征点被当作对比度不足的点滤除 
    double thresh_edge = 10; //大于此阈值的特征点被当作边缘点滤除 

    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(0, 3, thresh_contrast, thresh_edge, 1.6);
    //cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create(10000);

    f2d->detect(image, keypoints);

    //Drawing is not used in matching
    cv::drawKeypoints(image_rgb, keypoints, image_rgb, cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::namedWindow("2D Features", 1);
    cv::imshow("2D Features", image_rgb);

    cv::waitKey(0);

    return 0;
}

int siftMatch(std::string path_left, std::string path_right)
{
    cv::Mat image_left, image_left_rgb = cv::imread(path_left, cv::IMREAD_COLOR);
    cv::Mat image_right, image_right_rgb = cv::imread(path_right, cv::IMREAD_COLOR);

    if (image_left_rgb.empty())
    {
        std::cout << "Fail to read the image: " << path_left << std::endl;
        return -1;
    }
    if (image_right_rgb.empty())
    {
        std::cout << "Fail to read the image: " << path_right << std::endl;
        return -1;
    }

    cv::imshow("Left Image", image_left_rgb);
    cv::imshow("Right Image", image_right_rgb);
    cv::waitKey(0);

    cv::cvtColor(image_left_rgb, image_left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_right_rgb, image_right, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints_left, keypoints_right;

    cv::Ptr<cv::FeatureDetector> f2d = cv::xfeatures2d::SIFT::create(0, 3, 0.05, 10, 1.6);

    f2d->detect(image_left, keypoints_left);
    f2d->detect(image_right, keypoints_right);

    //Draw key points
    cv::drawKeypoints(image_left_rgb, keypoints_left, image_left_rgb, cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(image_right_rgb, keypoints_right, image_right_rgb, cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Left Features", image_left_rgb);
    cv::imshow("Right Features", image_right_rgb);

    //compute HOG(Histogram of Oriented Gradients) of keypoints
    cv::Mat hog_left, hog_right;

    f2d->compute(image_left, keypoints_left, hog_left);
    f2d->compute(image_right, keypoints_right, hog_right);

    //matching of the feature vector
    //cv::BFMatcher matcher; //"Burte Force" matcher, search all possible pairs, higer precision but much slower.
    cv::FlannBasedMatcher matcher; //Fast Library for Approximate Nearest Neighbors, lower precision but much faster.

    std::vector<cv::DMatch> matches;
    matcher.match(hog_left, hog_right, matches);

    //draw the raw matching result
    cv::Mat raw_match_image;
    cv::drawMatches(image_left, keypoints_left, image_right, keypoints_right, matches, raw_match_image);

    cv::imshow("Raw Matches", raw_match_image);
    cv::waitKey(0);

    //remove the bad matches, compute the Euclid distance of two matched vectors
    double max_distance = matches[0].distance, min_distance = matches[0].distance;

    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_distance) min_distance = dist;
        if (dist > max_distance) max_distance = dist;
    }

    std::cout << "---The worst match distance:" << max_distance << std::endl;
    std::cout << "---The best match distance:" << min_distance << std::endl;
    
    //draw the good matches, distance < 3*min_dist
    
    std::vector<cv::DMatch> good_matches;

    for (int  i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 3 * min_distance)
            good_matches.push_back(matches[i]);
    }

    std::cout << "The number of good matches:" << good_matches.size() << std::endl;

    cv::Mat good_matches_image;
    cv::drawMatches(image_left, keypoints_left, image_right, keypoints_right, good_matches, good_matches_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Good Matches", good_matches_image);
    cv::waitKey(0);

    if (good_matches.size() < 8)
    {
        std::cout << "There are no enough good matches! HOMOGRAPHY MATRIX cannot be found!" << std::endl;
        return -1;
    }

    //compute the homography matrix and locate the corresponding area of left image
    std::vector<cv::Point2f> corr_point_left;
    std::vector<cv::Point2f> corr_point_right;

    for (int i = 0; i < good_matches.size(); i++)
    {
        corr_point_left.push_back(keypoints_left[good_matches[i].queryIdx].pt);
        corr_point_right.push_back(keypoints_right[good_matches[i].trainIdx].pt);
    }

    //find the homography matrix using RANSAC, only work for flann transformation
    cv::Mat perspective_mat = cv::findHomography(corr_point_left, corr_point_right, cv::RANSAC);

    //draw the final result
    //the corners of the left image
    std::vector<cv::Point2f> corners_left(4);
    corners_left[0] = cv::Point2f(0, 0);
    corners_left[1] = cv::Point2f(image_left.cols, 0);
    corners_left[2] = cv::Point2f(image_left.cols, image_left.rows);
    corners_left[3] = cv::Point2f(0, image_left.rows);

    //compute the corresponding corners of the left image in the right one
    std::vector<cv::Point2f> corners_left_in_right;
    cv::perspectiveTransform(corners_left, corners_left_in_right, perspective_mat);

    //draw object
    cv::line(good_matches_image, corners_left_in_right[0] + cv::Point2f(image_left.cols, 0), corners_left_in_right[1] + cv::Point2f(image_left.cols, 0), cv::Scalar(0, 255, 0), 2);
    cv::line(good_matches_image, corners_left_in_right[1] + cv::Point2f(image_left.cols, 0), corners_left_in_right[2] + cv::Point2f(image_left.cols, 0), cv::Scalar(0, 255, 0), 2);
    cv::line(good_matches_image, corners_left_in_right[2] + cv::Point2f(image_left.cols, 0), corners_left_in_right[3] + cv::Point2f(image_left.cols, 0), cv::Scalar(0, 255, 0), 2);
    cv::line(good_matches_image, corners_left_in_right[3] + cv::Point2f(image_left.cols, 0), corners_left_in_right[0] + cv::Point2f(image_left.cols, 0), cv::Scalar(0, 255, 0), 2);

    cv::imshow("Match Result", good_matches_image);
    cv::waitKey(0);

    return 0;
}

int harrisDetector(std::string filename)
{
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

int main()
{
    std::string filename = "Cameraman.png";
    //harrisDetector(filename);
    //siftDetector(filename);
    std::string path_left_image = "../image/tail.bmp";
    //std::string path_right_image = "../image/tail2.bmp";
    std::string path_right_image = "../image/tail3.bmp";
    //std::string path_right_image = "../image/tail4.bmp";

    siftMatch(path_left_image, path_right_image);

    return 0;
}

