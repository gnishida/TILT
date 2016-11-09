#include "TILT.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

int main() {
	/*
	cv::Mat_<double> A = (cv::Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	cv::GaussianBlur(A, A, cv::Size(3, 3), 3, 3, cv::BORDER_CONSTANT);
	std::cout << A << std::endl;
	*/




	//cv::Mat img = cv::imread("../testdata/Facade_2_47385.png");	// soso
	//cv::Mat img = cv::imread("../testdata/Facade_1_86634.png");	// good
	//cv::Mat img = cv::imread("../testdata/Facade_4_47401.png");	// slow
	cv::Mat img = cv::imread("../testdata/Facade_3_392888.png");	// soso
	//cv::Mat img = cv::imread("../testdata/test.png"); 
	cv::Mat result;
	tilt::tilt(img, result);

	cv::imwrite("result.png", result);
	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
	cv::imshow("result", result);
	cv::waitKey(0);

	return 0;
}