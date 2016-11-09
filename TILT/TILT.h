#pragma once

#include "opencv2/core.hpp"

namespace tilt {

	void tilt(cv::Mat input_image, cv::Mat& result_image);
	void tilt_kernel(cv::Mat input_image, cv::Point2d center, cv::Size focus_size, cv::Mat initial_tfm_matrix, double outer_tol, int outer_max_iter, double inner_tol, cv::Mat& Dotau, double& f, cv::Mat_<double>& tfm_matrix);
	cv::Point2d transform_point(cv::Point2d pt, cv::Mat_<double> tfm_matrix);
	void imtransform_around_point(cv::Mat input_image, cv::Mat tfm_matrix, cv::Point2d point, cv::Mat& output_image);
	void tfm2para(cv::Mat_<double> tfm_matrix, cv::Size focus_size, cv::Point2d center, cv::Mat_<double>& tau);
	void para2tfm(cv::Mat_<double> tau, cv::Size focus_size, cv::Point2d center, cv::Mat_<double>& tfm_matrix);
	void jacobi(cv::Mat_<double> du, cv::Mat_<double> dv, cv::Size focus_size, cv::Point2d center, cv::Mat_<double> tau, cv::Mat& J);
	double inner_IALM_constraints(cv::Mat Dotau, cv::Mat J, cv::Mat S_J, double inner_tol, cv::Mat& A, cv::Mat& E, cv::Mat& delta_tau);
	void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat& X, cv::Mat& Y);
	void constraints(cv::Size focus_size, cv::Mat_<double> tau, cv::Mat_<double>& S);
	void svd(cv::Mat A, cv::Mat& U, cv::Mat& S, cv::Mat& VT);
	void mat_compare(cv::Mat src, double threshold, cv::Mat& dst, int cmpop);
	int rank(cv::Mat mat);
	double norm2(cv::Mat mat);
}

