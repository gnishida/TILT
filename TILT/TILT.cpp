#include "TILT.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace tilt {

	void tilt(cv::Mat args_input_image, cv::Mat& result_image) {
		// default value
		bool blur = true;
		int blur_kernel_size_k = 2;
		int blur_kernel_sigma_k = 2;
		cv::Mat_<double> initial_tfm_matrix = cv::Mat_<double>::eye(3, 3);
		int branch_accuracy = 5;
		double branch_max_rotation = CV_PI / 6.0;
		double branch_max_skew = 1.0;
		bool pyramid = true;
		int pyramid_max_level = 2;
		double outer_tol = 1e-4;
		double outer_tol_step = 10;
		int outer_max_iter = 50;
		double inter_tol = 1e-4;

		// down-sample the image if the focus is too large
		cv::Mat_<double> pre_scale_matrix = cv::Mat_<double>::eye(3, 3);
		double focus_threshold = 200.0;
		int min_length = std::min(args_input_image.rows, args_input_image.cols);
		if (min_length > focus_threshold) {
			double s = min_length / focus_threshold;
			cv::Size dst_pt_size = cv::Size(args_input_image.cols / s, args_input_image.rows / s);
			cv::resize(args_input_image, args_input_image, dst_pt_size);
			pre_scale_matrix = (cv::Mat_<double>(3, 3) << s, 0, 0, 0, s, 0, 0, 0, 1);
		}

		cv::Point2d args_center(args_input_image.cols * 0.5, args_input_image.rows * 0.5);
		cv::Size focus_size(args_input_image.cols, args_input_image.rows);

		// step 1: prepare data for the lowest resolution
		focus_threshold = 50.0;
		int total_scale = std::floor(std::log2(std::min(focus_size.width, focus_size.height) / focus_threshold));
		cv::Mat_<double> downsample_matrix = (cv::Mat_<double>(3, 3) << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 1);
		cv::Mat_<double> scale_matrix;
		cv::pow(downsample_matrix, total_scale, scale_matrix);

		cv::Mat input_image;
		if (blur) {
			cv::GaussianBlur(args_input_image, input_image, cv::Size(std::ceil(blur_kernel_size_k * pow(2, total_scale)) + 1, std::ceil(blur_kernel_size_k * pow(2, total_scale)) + 1), std::ceil(blur_kernel_sigma_k * pow(2, total_scale)) + 1, std::ceil(blur_kernel_sigma_k * pow(2, total_scale)) + 1, cv::BORDER_CONSTANT);
		}
		else {
			input_image = args_input_image.clone();
		}

		cv::warpPerspective(input_image, input_image, scale_matrix, cv::Size(focus_size.width * pow(0.5, total_scale), focus_size.height * pow(0.5, total_scale)));

		if (input_image.channels() > 1) {
			cv::cvtColor(input_image, input_image, cv::COLOR_BGR2GRAY);
		}

		input_image.convertTo(input_image, CV_64F);

		initial_tfm_matrix = scale_matrix * initial_tfm_matrix * scale_matrix.inv();
		cv::Point2d center = transform_point(args_center, scale_matrix);
		center = cv::Point2d(std::floor(center.x), std::floor(center.y));

		std::vector<std::vector<double>> f_branch(3, std::vector<double>(2 * branch_accuracy + 1, 0));
		std::vector<std::vector<cv::Mat>> Dotau_branch(3, std::vector<cv::Mat>(2 * branch_accuracy + 1));
		std::vector<std::vector<cv::Mat>> result_tfm_matrix(3, std::vector<cv::Mat>(2 * branch_accuracy + 1));

		// step 2: design branch-and-bound method.
		std::vector<std::vector<cv::Mat_<double>>> candidate_matrix(3, std::vector<cv::Mat_<double>>(2 * branch_accuracy + 1));
		for (int i = 0; i < 2 * branch_accuracy + 1; ++i) {
			candidate_matrix[0][i] = cv::Mat_<double>::eye(3, 3);
			double theta = -branch_max_rotation + i * branch_max_rotation / branch_accuracy;
			candidate_matrix[0][i](0, 0) = cos(theta);
			candidate_matrix[0][i](0, 1) = -sin(theta);
			candidate_matrix[0][i](1, 0) = sin(theta);
			candidate_matrix[0][i](1, 1) = cos(theta);
			//candidate_matrix[0][i] = trans_matrix2 * candidate_matrix[0][i] * trans_matrix1;
			candidate_matrix[1][i] = cv::Mat_<double>::eye(3, 3);
			candidate_matrix[1][i](0, 1) = -branch_max_skew + i * branch_max_skew / branch_accuracy;
			//candidate_matrix[1][i] = trans_matrix2 * candidate_matrix[1][i] * trans_matrix1;
			candidate_matrix[2][i] = cv::Mat_<double>::eye(3, 3);
			candidate_matrix[2][i](1, 0) = -branch_max_skew + i * branch_max_skew / branch_accuracy;
			//candidate_matrix[2][i] = trans_matrix2 * candidate_matrix[2][i] * trans_matrix1;
		}

		// step 3: begin branch-and-bound
		int gap = 5;
		int level = 3;
		//cv::Mat_<double> trans_matrix1 = (cv::Mat_<double>(3, 3) << 1, 0, -center.x, 0, 1, -center.y, 0, 0, 1);
		//cv::Mat_<double> trans_matrix2 = (cv::Mat_<double>(3, 3) << 1, 0, center.x, 0, 1, center.y, 0, 0, 1);
		cv::Mat BLACK_MATRIX(input_image.rows * level + gap * (level - 1), input_image.cols * (2 * branch_accuracy + 1) + gap * 2 * branch_accuracy, CV_8U, cv::Scalar(0));
		for (int i = 0; i < level; ++i) {
			for (int j = 0; j < 2 * branch_accuracy + 1; ++j) {
				//cv::Mat tfm_matrix = (candidate_matrix[i][j] * initial_tfm_matrix.inv()).inv();
				cv::Mat tfm_matrix = candidate_matrix[i][j] * initial_tfm_matrix.inv();

				//tfm_matrix = trans_matrix2 * tfm_matrix * trans_matrix1;

				cv::Mat Dotau;
				//cv::warpPerspective(input_image, Dotau, tfm_matrix, cv::Size(input_image.cols, input_image.rows));
				imtransform_around_point(input_image, tfm_matrix, center, Dotau);

				// copy the transformed image to BLACK_MATRIX
				Dotau.copyTo(BLACK_MATRIX(cv::Rect((input_image.cols + gap) * j, (input_image.rows + gap) * i, input_image.cols, input_image.rows)));
				//cv::Mat roi = BLACK_MATRIX(cv::Rect((input_image.cols + gap) * j, (input_image.rows + gap) * i, input_image.cols, input_image.rows));
				//Dotau.copyTo(roi);

				Dotau.convertTo(Dotau, CV_64F);
				Dotau = Dotau / cv::norm(Dotau);
				cv::Mat S, U, V;
				cv::SVD::compute(Dotau, S, U, V);
				double f = cv::sum(S)[0];

				// copy the transformation data
				f_branch[i][j] = f;
				Dotau_branch[i][j] = Dotau;
				result_tfm_matrix[i][j] = tfm_matrix.inv();
			}

			int index = std::distance(f_branch[i].begin(), std::min_element(f_branch[i].begin(), f_branch[i].end()));
			initial_tfm_matrix = result_tfm_matrix[i][index];
		}

		// step 4: adapt initial_tfm_matrix to highest-resolution
		initial_tfm_matrix = scale_matrix.inv() * initial_tfm_matrix * scale_matrix;

		// step 5: show inter result if necessary
		cv::namedWindow("98", cv::WINDOW_AUTOSIZE);
		cv::imshow("98", BLACK_MATRIX);
		cv::waitKey(0);

		// Do pyramid if necessary
		if (pyramid) {
			// define parameters
			downsample_matrix = (cv::Mat_<double>(3, 3) << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 1);
			cv::Mat_<double> upsample_matrix = downsample_matrix.inv();
			total_scale = std::ceil(std::max(std::log2(std::min(focus_size.width, focus_size.height) / focus_threshold), 0.0));

			for (int scale = total_scale; scale >= 0; --scale) {
				if (total_scale - scale >= pyramid_max_level) break;

				// blur if required
				if (blur && scale > 0) {
					cv::GaussianBlur(args_input_image, input_image, cv::Size(std::ceil(blur_kernel_size_k * pow(2, scale)) + 1, std::ceil(blur_kernel_size_k * pow(2, scale)) + 1), std::ceil(blur_kernel_sigma_k * pow(2, scale)) + 1, std::ceil(blur_kernel_sigma_k * pow(2, scale)) + 1, cv::BORDER_CONSTANT);
				}
				else {
					input_image = args_input_image.clone();
				}

				// prepare image and initial tfm_matrix
				cv::pow(downsample_matrix, scale, scale_matrix);

				cv::warpPerspective(input_image, input_image, scale_matrix, cv::Size(input_image.cols * pow(0.5, scale), input_image.rows * pow(0.5, scale)));
				cv::Mat_<double> tfm_matrix = scale_matrix * initial_tfm_matrix * scale_matrix.inv();
				cv::Point2d center = transform_point(args_center, scale_matrix);
				center = cv::Point2d(floor(center.x), floor(center.y));
				focus_size = cv::Size(input_image.cols, input_image.rows);

				cv::Mat Dotau;
				double f;
				cv::Mat updated_tfm_matrix;
				tilt_kernel(input_image, center, focus_size, tfm_matrix, outer_tol, outer_max_iter, inter_tol, Dotau, f, tfm_matrix);

				// update tfm_matrix of the highest-resolution level
				initial_tfm_matrix = scale_matrix.inv() * tfm_matrix * scale_matrix;
				outer_tol = outer_tol * outer_tol_step;
			}

			//cv::Mat tfm_matrix = initial_tfm_matrix;
		}

		// transform the result back to the original image before incising and resizing
		initial_tfm_matrix = pre_scale_matrix * initial_tfm_matrix * pre_scale_matrix.inv();
		std::cout << pre_scale_matrix << std::endl;
		std::cout << initial_tfm_matrix << std::endl;

		imtransform_around_point(args_input_image, initial_tfm_matrix, cv::Point2d(args_input_image.cols * 0.5, args_input_image.rows * 0.5), result_image);
	}

	void tilt_kernel(cv::Mat input_image, cv::Point2d center, cv::Size focus_size, cv::Mat initial_tfm_matrix, double outer_tol, int outer_max_iter, double inner_tol, cv::Mat& Dotau, double& f, cv::Mat_<double>& tfm_matrix) {
		if (input_image.channels() > 1) {
			std::vector<cv::Mat> channels;
			cv::split(input_image, channels);

			input_image = cv::Mat(input_image.size(), CV_8U);
			input_image = channels[2] * 0.299 + channels[1] * 0.587 + channels[0] * 0.144;
		}

		input_image.convertTo(input_image, CV_64F);

		// make base_points and focus_size integer, the effect of this operations remains to be tested
		//center = cv::Point2d(std::floor(center.x), std::floor(center.y));
		//focus_size = cv::Size(std::floor(focus_size.width), std::floor(focus_size.height));

		// decide origin of the two axes

		// prepare initial data
		cv::Mat input_du;
		cv::Sobel(input_image, input_du, CV_64F, 1, 0, 3, 1.0 / 8.0, 0, cv::BORDER_CONSTANT);
		cv::Mat input_dv;
		cv::Sobel(input_image, input_dv, CV_64F, 0, 1, 3, 1.0 / 8.0, 0, cv::BORDER_CONSTANT);
		tfm_matrix = initial_tfm_matrix;

		imtransform_around_point(input_image, tfm_matrix.inv(), center, Dotau);
		cv::Mat initial_image = Dotau.clone();

		cv::Mat du;
		imtransform_around_point(input_du, tfm_matrix.inv(), center, du);
		cv::Mat dv;
		imtransform_around_point(input_dv, tfm_matrix.inv(), center, dv);

		cv::Mat tmp;
		cv::multiply(Dotau, du, tmp);
		du = du / cv::norm(Dotau) - cv::sum(tmp)[0] / pow(cv::norm(Dotau), 3) * Dotau;
		cv::multiply(Dotau, dv, tmp);
		dv = dv / cv::norm(Dotau) - cv::sum(tmp)[0] / pow(cv::norm(Dotau), 3) * Dotau;
		double A_scale = cv::norm(Dotau);
		Dotau = Dotau / A_scale;
		
		cv::Mat_<double> tau;
		tfm2para(tfm_matrix, focus_size, tau);

		cv::Mat_<double> J, S;
		jacobi(du, dv, focus_size, tau, J);
		std::vector<cv::Mat> J_channels;
		cv::split(J, J_channels);
		constraints(focus_size, tau, S);
				
		double pre_f = 0.0;

		// begin main loop
		for (int outer_round = 0; ; ++outer_round) {
			cv::Mat_<double> delta_tau;
			cv::Mat A, E;
			f = inner_IALM_constraints(Dotau, J, S, inner_tol, A, E, delta_tau);

			// display information
			std::cout << "outer_round " << outer_round + 1 << ", f=" << f << ", rank(A) =" << rank(A) << ", ||E||_1=" << cv::sum(cv::abs(E))[0] << std::endl;

			// update Dotau
			tau = tau + delta_tau;
			para2tfm(tau, focus_size, tfm_matrix);
			imtransform_around_point(input_image, tfm_matrix.inv(), center, Dotau);

			// judge convergence
			if (outer_round >= outer_max_iter || abs(f - pre_f) < outer_tol) break;

			// record data and prepare for the next round
			pre_f = f;
			imtransform_around_point(input_du, tfm_matrix.inv(), center, du);
			imtransform_around_point(input_dv, tfm_matrix.inv(), center, dv);

			cv::Mat tmp;
			cv::multiply(Dotau, du, tmp);
			du = du / cv::norm(Dotau) - cv::sum(tmp)[0] / pow(cv::norm(Dotau), 3) * Dotau;
			cv::multiply(Dotau, dv, tmp);
			dv = dv / cv::norm(Dotau) - cv::sum(tmp)[0] / pow(cv::norm(Dotau), 3) * Dotau;
			double A_scale = cv::norm(Dotau);
			Dotau = Dotau / A_scale;
			jacobi(du, dv, focus_size, tau, J);
			constraints(focus_size, tau, S);
		}

		// display results
		cv::namedWindow("102", cv::WINDOW_AUTOSIZE);
		double Dotau_min, Dotau_max;
		cv::minMaxLoc(Dotau, &Dotau_min, &Dotau_max);
		cv::imshow("102", Dotau / Dotau_max);
		cv::waitKey(0);
	}

	cv::Point2d transform_point(cv::Point2d pt, cv::Mat_<double> tfm_matrix) {
		cv::Mat_<double> ret = tfm_matrix * (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1);
		return cv::Point2d(ret(0, 0) / ret(2, 0), ret(1, 0) / ret(2, 0));
	}

	void imtransform_around_point(cv::Mat input_image, cv::Mat tfm_matrix, cv::Point2d point, cv::Mat& output_image) {
		cv::Mat_<double> trans_matrix1 = (cv::Mat_<double>(3, 3) << 1, 0, -point.x, 0, 1, -point.y, 0, 0, 1);
		cv::Mat_<double> trans_matrix2 = (cv::Mat_<double>(3, 3) << 1, 0, point.x, 0, 1, point.y, 0, 0, 1);

		cv::warpPerspective(input_image, output_image, trans_matrix2 * tfm_matrix * trans_matrix1, cv::Size(input_image.cols, input_image.rows));
	}

	void tfm2para(cv::Mat_<double> tfm_matrix, cv::Size focus_size, cv::Mat_<double>& tau) {
		cv::Point2d center(focus_size.width * 0.5, focus_size.height * 0.5);
		//cv::Point2d center(std::floor((focus_size.width - 1) * 0.5), std::floor((focus_size.height - 1) * 0.5));
		cv::Mat_<double> pts = (cv::Mat_<double>(3, 4) << -center.x, focus_size.width - 1 - center.x, focus_size.width - 1 - center.x, -center.x,
			-center.y, -center.y, focus_size.height - 1 - center.y, focus_size.height - 1 - center.y, 1, 1, 1, 1);
		pts = tfm_matrix * pts;

		// 変換パラメータτは、画像の矩形の4つの頂点の座標で表現できる
		tau = cv::Mat_<double>(8, 1);
		tau(0, 0) = pts(0, 0) / pts(2, 0);
		tau(1, 0) = pts(1, 0) / pts(2, 0);
		tau(2, 0) = pts(0, 1) / pts(2, 1);
		tau(3, 0) = pts(1, 1) / pts(2, 1);
		tau(4, 0) = pts(0, 2) / pts(2, 2);
		tau(5, 0) = pts(1, 2) / pts(2, 2);
		tau(6, 0) = pts(0, 3) / pts(2, 3);
		tau(7, 0) = pts(1, 3) / pts(2, 3);
	}

	/**
	 * ...
	 * 
	 * @param tau			column vector of tau
	 * @param focus_size
	 * @param tfm_matrix
	 */
	void para2tfm(cv::Mat_<double> tau, cv::Size focus_size, cv::Mat_<double>& tfm_matrix) {
		cv::Point2d center(focus_size.width * 0.5, focus_size.height * 0.5);
		//cv::Point2d center(std::floor((focus_size.width - 1) * 0.5), std::floor((focus_size.height - 1) * 0.5));

		cv::Mat_<double> X = (cv::Mat_<double>(1, 4) << -center.x, focus_size.width - 1 - center.x, focus_size.width - 1 - center.x, -center.x);
		cv::Mat_<double> Y = (cv::Mat_<double>(1, 4) << -center.y, -center.y, focus_size.height - 1 - center.y, focus_size.height - 1 - center.y);

		cv::Mat_<double> A = cv::Mat_<double>::zeros(8, 8);
		cv::Mat_<double> b = cv::Mat_<double>::zeros(8, 1);

		for (int i = 0; i < 4; ++i) {
			A(2 * i, 0) = 0;
			A(2 * i, 1) = 0;
			A(2 * i, 2) = 0;
			A(2 * i, 3) = -X(0, i);
			A(2 * i, 4) = -Y(0, i);
			A(2 * i, 5) = -1;
			A(2 * i, 6) = tau(i * 2 + 1, 0) * X(0, i);
			A(2 * i, 7) = tau(i * 2 + 1, 0) * Y(0, i);

			A(2 * i + 1, 0) = X(0, i);
			A(2 * i + 1, 1) = Y(0, i);
			A(2 * i + 1, 2) = 1;
			A(2 * i + 1, 3) = 0;
			A(2 * i + 1, 4) = 0;
			A(2 * i + 1, 5) = 0;
			A(2 * i + 1, 6) = -tau(i * 2, 0) * X(0, i);
			A(2 * i + 1, 7) = -tau(i * 2, 0) * Y(0, i);

			b(2 * i, 0) = -tau(i * 2 + 1, 0);
			b(2 * i + 1, 0) = tau(i * 2, 0);
		}

		cv::Mat_<double> solution = A.inv() * b;

		tfm_matrix = cv::Mat_<double>(3, 3);
		tfm_matrix(0, 0) = solution(0, 0);
		tfm_matrix(0, 1) = solution(1, 0);
		tfm_matrix(0, 2) = solution(2, 0);
		tfm_matrix(1, 0) = solution(3, 0);
		tfm_matrix(1, 1) = solution(4, 0);
		tfm_matrix(1, 2) = solution(5, 0);
		tfm_matrix(2, 0) = solution(6, 0);
		tfm_matrix(2, 1) = solution(7, 0);
		tfm_matrix(2, 2) = 1;
	}

	void jacobi(cv::Mat_<double> du, cv::Mat_<double> dv, cv::Size focus_size, cv::Mat_<double> tau, cv::Mat& J) {
		cv::Point2d center(std::floor(focus_size.width * 0.5), std::floor(focus_size.height * 0.5));
		//cv::Point2d center(std::floor((focus_size.width - 1) * 0.5), std::floor((focus_size.height - 1) * 0.5));

		int m = du.rows;
		int n = du.cols;

		cv::Mat X0, Y0;
		meshgrid(cv::Range(-center.x, focus_size.width - 1 - center.x), cv::Range(-center.y, focus_size.height - 1 - center.y), X0, Y0);

		cv::Mat_<double> H;
		para2tfm(tau, focus_size, H);
		cv::Mat_<double> N1 = H(0, 0)*X0 + H(0, 1)*Y0 + H(0, 2);
		cv::Mat_<double> N2 = H(1, 0)*X0 + H(1, 1)*Y0 + H(1, 2);
		cv::Mat_<double> N = H(2, 0)*X0 + H(2, 1)*Y0 + 1;
		cv::Mat_<double> N_squared;
		cv::pow(N, 2, N_squared);

		//std::vector<int> sizes = { m, n, 8 };

		cv::Mat dIdH;// = cv::Mat(3, sizes.data(), CV_64FC(1), cv::Scalar::all(0));
		std::vector<cv::Mat_<double>> dIdH_channels(8);
		cv::multiply(du, X0, dIdH_channels[0]);
		cv::divide(dIdH_channels[0], N, dIdH_channels[0]);
		cv::multiply(du, Y0, dIdH_channels[1]);
		cv::divide(dIdH_channels[1], N, dIdH_channels[1]);
		cv::divide(du, N, dIdH_channels[2]);
		cv::multiply(dv, X0, dIdH_channels[3]);
		cv::divide(dIdH_channels[3], N, dIdH_channels[3]);
		cv::multiply(dv, Y0, dIdH_channels[4]);
		cv::divide(dIdH_channels[4], N, dIdH_channels[4]);
		cv::divide(dv, N, dIdH_channels[5]);

		cv::Mat_<double> tmp1, tmp2;

		cv::divide(-N1, N_squared, tmp1);
		cv::multiply(tmp1, X0, tmp1);
		cv::multiply(du, tmp1, tmp1);
		cv::divide(-N2, N_squared, tmp2);
		cv::multiply(tmp2, X0, tmp2);
		cv::multiply(dv, tmp2, tmp2);
		dIdH_channels[6] = tmp1 + tmp2;

		cv::divide(-N1, N_squared, tmp1);
		cv::multiply(tmp1, Y0, tmp1);
		cv::multiply(du, tmp1, tmp1);
		cv::divide(-N2, N_squared, tmp2);
		cv::multiply(tmp2, Y0, tmp2);
		cv::multiply(dv, tmp2, tmp2);
		dIdH_channels[7] = tmp1 + tmp2;

		cv::merge(dIdH_channels, dIdH);

		cv::Mat_<double> dPdH = cv::Mat_<double>::zeros(8, 8);
		cv::Mat_<double> X = (cv::Mat_<double>(1, 4) << -center.x, focus_size.width - 1 - center.x, focus_size.width - 1 - center.x, -center.x);
		cv::Mat_<double> Y = (cv::Mat_<double>(1, 4) << -center.y, -center.y, focus_size.height - 1 - center.y, focus_size.height - 1 - center.y);
		N1 = X * H(0, 0) + Y * H(0, 1) + H(0, 2);
		N2 = X * H(1, 0) + Y * H(1, 1) + H(1, 2);
		N = X * H(2, 0) + Y * H(2, 1) + 1;
		for (int i = 0; i < 4; ++i) {
			dPdH(i * 2, 0) = X(0, i) / N(0, i);
			dPdH(i * 2, 1) = Y(0, i) / N(0, i);
			dPdH(i * 2, 2) = 1.0 / N(0, i);
			dPdH(i * 2, 6) = -N1(0, i) / N(0, i) / N(0, i) * X(0, i);
			dPdH(i * 2, 7) = -N1(0, i) / N(0, i) / N(0, i) * Y(0, i);
			dPdH(i * 2 + 1, 3) = X(0, i) / N(0, i);
			dPdH(i * 2 + 1, 4) = Y(0, i) / N(0, i);
			dPdH(i * 2 + 1, 5) = 1.0 / N(0, i);
			dPdH(i * 2 + 1, 6) = -N2(0, i) / N(0, i) / N(0, i) * X(0, i);
			dPdH(i * 2 + 1, 7) = -N2(0, i) / N(0, i) / N(0, i) * Y(0, i);
		}
		cv::Mat_<double> dHdP = dPdH.inv();

		std::vector<cv::Mat> J_channels(8);
		for (int i = 0; i < 8; ++i) {
			J_channels[i] = cv::Mat_<double>::zeros(m, n);
			for (int j = 0; j < 8; ++j) {
				J_channels[i] += dIdH_channels[j] * dHdP(j, i);
			}
		}

		cv::merge(J_channels, J);
	}

	double inner_IALM_constraints(cv::Mat Dotau, cv::Mat J, cv::Mat S_J, double inner_tol, cv::Mat& A, cv::Mat& E, cv::Mat& delta_tau) {
		double c = 1.0;
		double mu = 1.25 / cv::norm(Dotau);
		int max_iter = 999999;
		
		// prepare data
		int m = Dotau.rows;
		int n = Dotau.cols;
		E = cv::Mat_<double>::zeros(m, n);
		A = cv::Mat_<double>::zeros(m, n);
		int p = J.channels();
		delta_tau = cv::Mat_<double>::zeros(p, 1);

		cv::Mat J_vec = cv::Mat_<double>::zeros(m * n, p);
		std::vector<cv::Mat> J_channels;
		cv::split(J, J_channels);
		for (int i = 0; i < J_channels.size(); ++i) {
			cv::Mat roi(J_vec, cv::Rect(i, 0, 1, J_vec.rows));
			((cv::Mat)J_channels[i].t()).reshape(1, m * n).copyTo(roi);
		}
		
		

		cv::Mat Jo = J_vec.clone();
		J_vec.push_back(S_J);
		cv::Mat pinv_J_vec = J_vec.inv(cv::DECOMP_SVD);
		int inner_round = 0;
		double rho = 1.25;
		double lambda = c / sqrt(m);

		cv::Mat Y_1 = Dotau.clone();
		double norm_two = cv::norm(Y_1);
		double norm_inf = cv::norm(Y_1, cv::NORM_INF) / lambda;
		double dual_norm = std::max(norm_two, norm_inf);
		Y_1 = Y_1 / dual_norm;
		cv::Mat Y_2 = cv::Mat_<double>::zeros(S_J.rows, 1);
		double d_norm = cv::norm(Dotau);
		int error_sign = 0;

		cv::Mat S, U, VT;
		cv::SVD::compute(Dotau, S, U, VT);
		double first_f = cv::sum(S)[0];

		// begin main loop
		for (int inner_round = 0; inner_round < max_iter; ++inner_round) {
			cv::Mat temp_0 = Dotau + ((cv::Mat)(Jo * delta_tau).t()).reshape(1, m) + Y_1 / mu;
			cv::Mat temp_1 = temp_0 - E;
			cv::Mat S, U, VT;
			svd(temp_1, U, S, VT);

			cv::Mat S_temp;
			mat_compare(S, 1.0 / mu, S_temp, cv::CMP_GT);
			cv::multiply(S_temp, S - 1.0 / mu, S_temp);
			A = U * S_temp * VT;

			cv::Mat temp_2 = temp_0 - A;
			
			cv::Mat temp_2_temp;
			mat_compare(temp_2, lambda / mu, temp_2_temp, cv::CMP_GT);
			cv::multiply(temp_2_temp, temp_2 - lambda / mu, temp_2_temp);
			cv::Mat temp_2_temp2;
			mat_compare(temp_2, -lambda / mu, temp_2_temp2, cv::CMP_LT);
			cv::multiply(temp_2_temp2, temp_2 + lambda / mu, temp_2_temp2);
			E = temp_2_temp + temp_2_temp2;

			mat_compare(S, 1.0 / mu, S_temp, cv::CMP_GT);
			cv::multiply(S_temp, S - 1.0 / mu, S_temp);
			double f = cv::sum(cv::abs(S_temp))[0] + lambda * cv::sum(cv::abs(E))[0];

			cv::Mat temp_3 = A + E - Dotau - Y_1 / mu;
			temp_3 = ((cv::Mat)temp_3.t()).reshape(1, m * n);
			temp_3.push_back((cv::Mat)(-Y_2 / mu));

			delta_tau = pinv_J_vec * temp_3;
			//std::cout << delta_tau << std::endl;
			cv::Mat derivative_Y_1 = Dotau - A - E + ((cv::Mat)(Jo * delta_tau).t()).reshape(1, m);
			cv::Mat derivative_Y_2 = S_J * delta_tau;
			Y_1 = Y_1 + derivative_Y_1 * mu;
			Y_2 = Y_2 + derivative_Y_2 * mu;

			// judge error
			if (f < first_f / 3) {
				error_sign = 1;
				A = Dotau;
				E = 0;
				f = first_f;
				delta_tau = cv::Mat_<double>::zeros(J.channels(), 1);
				return f;
			}

			double stop_criterion = sqrt(pow(cv::norm(derivative_Y_1), 2) + pow(cv::norm(derivative_Y_2), 2)) / d_norm;
			//std::cout << cv::norm(derivative_Y_1) << ", " << cv::norm(derivative_Y_2) << ", " << stop_criterion << std::endl;
			if (stop_criterion < inner_tol) {
				return f;
			}

			mu *= rho;
		}
	}

	void meshgrid(const cv::Range &xrange, const cv::Range &yrange, cv::Mat& X, cv::Mat& Y) {
		std::vector<double> t_x, t_y;
		for (int i = xrange.start; i <= xrange.end; i++) t_x.push_back(i);
		for (int i = yrange.start; i <= yrange.end; i++) t_y.push_back(i);

		cv::Mat xgv = cv::Mat(t_x);
		cv::Mat ygv = cv::Mat(t_y);

		cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
		cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
	}

	void constraints(cv::Size focus_size, cv::Mat_<double> tau, cv::Mat_<double>& S) {
		S = cv::Mat_<double>::zeros(1, 8);

		cv::Mat_<double> e1 = (cv::Mat_<double>(2, 1) << tau(4, 0) - tau(0, 0), tau(5, 0) - tau(1, 0));
		cv::Mat_<double> e2 = (cv::Mat_<double>(2, 1) << tau(6, 0) - tau(2, 0), tau(7, 0) - tau(3, 0));
		double norm_e1 = ((cv::Mat_<double>)(e1.t() * e1))(0, 0);
		double norm_e2 = ((cv::Mat_<double>)(e2.t() * e2))(0, 0);
		double e1e2 = ((cv::Mat_<double>)(e1.t() * e2))(0, 0);
		double N = 2 * sqrt(norm_e1 * norm_e2 - pow(e1e2, 2));

		S(0, 0) = 1.0 / N * (-2 * (tau(4, 0) - tau(0, 0)) * norm_e2 + 2 * e1e2 * (tau(6, 0) - tau(2, 0)));
		S(0, 1) = 1.0 / N * (-2 * (tau(5, 0) - tau(1, 0)) * norm_e2 + 2 * e1e2 * (tau(7, 0) - tau(3, 0)));
		S(0, 2) = 1.0 / N * (-2 * (tau(6, 0) - tau(2, 0)) * norm_e1 + 2 * e1e2 * (tau(4, 0) - tau(0, 0)));
		S(0, 3) = 1.0 / N * (-2 * (tau(7, 0) - tau(3, 0)) * norm_e1 + 2 * e1e2 * (tau(5, 0) - tau(1, 0)));
		S(0, 4) = 1.0 / N * (2 * (tau(4, 0) - tau(0, 0)) * norm_e2 - 2 * e1e2 * (tau(6, 0) - tau(2, 0)));
		S(0, 5) = 1.0 / N * (2 * (tau(5, 0) - tau(1, 0)) * norm_e2 - 2 * e1e2 * (tau(7, 0) - tau(3, 0)));
		S(0, 6) = 1.0 / N * (2 * (tau(6, 0) - tau(2, 0)) * norm_e1 - 2 * e1e2 * (tau(4, 0) - tau(0, 0)));
		S(0, 7) = 1.0 / N * (2 * (tau(7, 0) - tau(3, 0)) * norm_e1 - 2 * e1e2 * (tau(5, 0) - tau(1, 0)));
	}

	void svd(cv::Mat A, cv::Mat& U, cv::Mat& S, cv::Mat& VT) {
		cv::Mat W;
		cv::SVD::compute(A, W, U, VT);

		S = cv::Mat::zeros(W.rows, W.rows, CV_64F);
		for (int i = 0; i < S.rows; ++i) {
			S.at<double>(i, i) = W.at<double>(i, 0);
		}
	}

	void mat_compare(cv::Mat src, double threshold, cv::Mat& dst, int cmpop) {
		cv::compare(src, threshold, dst, cmpop);
		dst.convertTo(dst, CV_64F);
		dst = dst / 255.0;
	}

	int rank(cv::Mat mat) {
		// Compute SVD
		cv::Mat w, u, vt;
		cv::SVD::compute(mat, w, u, vt);

		// w is the matrix of singular values
		// Find non zero singular values.

		// Use a small threshold to account for numeric errors
		cv::Mat nonZeroSingularValues = w > 0.0001;

		// Count the number of non zero
		return cv::countNonZero(nonZeroSingularValues);
	}

}
