#pragma once
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <ceres/ceres.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "../fit_superquadric_ceres.h"
#include "../superquadric_formulas.h"
#include <pcl/visualization/pcl_visualizer.h>


////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename MatScalar>
sq::SuperquadricFittingCeres<PointT, MatScalar>::SuperquadricFittingCeres()
	: pre_align_(true)
	, pre_align_axis_(2)
{
	init_parameters_.a = 1.;
	init_parameters_.b = 1.;
	init_parameters_.c = 1.;
	init_parameters_.e1 = 1.;
	init_parameters_.e2 = 1.;
	init_parameters_.transform = Eigen::Matrix4d::Identity();
}

//////////////////////////////////////////////////////////////////////////////////

//
//
//
//////////////////////////////////////////////////////////////////////////////////
//template<typename PointT, typename MatScalar> double
//sq::SuperquadricFittingCeres<PointT, MatScalar>::fit(SuperquadricParameters<MatScalar> &parameters)
//{
//	Eigen::Matrix<MatScalar, 4, 4> transformation_prealign(Eigen::Matrix<MatScalar, 4, 4>::Identity());
//	Eigen::Matrix<MatScalar, 3, 1> variances;
//	variances(0) = variances(1) = variances(2) = static_cast <MatScalar> (1.);
//
//
//	preAlign(transformation_prealign, variances);
//	input_prealigned_.reset(new Cloud());
//	pcl::transformPointCloud(*input_, *input_prealigned_, transformation_prealign);
//
//
//	ceres::Problem problem;
//	const double doubleMax = std::numeric_limits<double>::max();
//	double xvec[11];
//	xvec[0] = xvec[1] = 1.;
//	xvec[2] = variances(0) * 3.;
//	xvec[3] = variances(1) * 3.;
//	xvec[4] = variances(2) * 3.;
//	xvec[5] = xvec[6] = xvec[7] = xvec[8] = xvec[9] = xvec[10] = 0.;
//
//
//
//	for (size_t p_i = 0; p_i < input_prealigned_->size(); ++p_i)
//	{
//		PointT &point = (*input_prealigned_)[p_i];
//		ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<SuperquadricCostFunctor, 1, 11>(new SuperquadricCostFunctor(point));
//		//    ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<SuperquadricCostFunctor, ceres::CENTRAL, 1, 11> (new SuperquadricCostFunctor (point));
//		problem.AddResidualBlock(cost_function, NULL, xvec);
//	}
//	problem.SetParameterLowerBound(xvec, 0, 0.1);
//	problem.SetParameterLowerBound(xvec, 1, 0.1);
//	problem.SetParameterLowerBound(xvec, 2, 0.00001);
//	problem.SetParameterLowerBound(xvec, 3, 0.00001);
//	problem.SetParameterLowerBound(xvec, 4, 0.00001);
//
//	problem.SetParameterUpperBound(xvec, 0, 2.01);
//	problem.SetParameterUpperBound(xvec, 1, 2.01);
//
//	ceres::Solver::Options options;
//	options.minimizer_type = ceres::TRUST_REGION;
//	//options.minimizer_type = ceres::LEVENBERG_MARQUARDT;
//	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//	options.minimizer_progress_to_stdout = false;
//	options.num_threads = 20;
//	options.max_num_iterations = 1000;
//
//	ceres::Solver::Summary summary;
//	ceres::Solve(options, &problem, &summary);
//
//
//	Eigen::Matrix<MatScalar, 4, 4> &transformation = parameters.transform;
//	transformation.setZero();
//	transformation(0, 3) = xvec[5];
//	transformation(1, 3) = xvec[6];
//	transformation(2, 3) = xvec[7];
//	transformation(3, 3) = 1.;
//	transformation.block(0, 0, 3, 3) = Eigen::AngleAxis<MatScalar>(xvec[8], Eigen::Matrix<MatScalar, 3, 1>::UnitZ()) *
//		Eigen::AngleAxis<MatScalar>(xvec[9], Eigen::Matrix<MatScalar, 3, 1>::UnitX()) *
//		Eigen::AngleAxis<MatScalar>(xvec[10], Eigen::Matrix<MatScalar, 3, 1>::UnitZ()).matrix();
//
//
//	//  clampParameters (xvec[0], xvec[1]);
//
//	parameters.e1 = xvec[0];
//	parameters.e2 = xvec[1];
//	parameters.a = xvec[2];
//	parameters.b = xvec[3];
//	parameters.c = xvec[4];
//	parameters.transform = Eigen::Matrix<MatScalar, 4, 4>(transformation) * transformation_prealign;
//
//	MatScalar final_error = computeSuperQuadricError<PointT, MatScalar>(input_, xvec[0], xvec[1], xvec[2], xvec[3], xvec[4], transformation);
//
//
//	return (final_error);
//}

template<typename PointT, typename MatScalar> void
sq::SuperquadricFittingCeres<PointT, MatScalar>::preAlign(Eigen::Matrix<MatScalar, 4, 4> &transformation_prealign,
	Eigen::Matrix<MatScalar, 3, 1> &variances)
{
	/// Compute the centroid
	Eigen::Vector4d centroid;
	pcl::compute3DCentroid(*input_, centroid);
	Eigen::Matrix<MatScalar, 4, 4> transformation_centroid(Eigen::Matrix<MatScalar, 4, 4>::Identity());
	transformation_centroid(0, 3) = -centroid(0);
	transformation_centroid(1, 3) = -centroid(1);
	transformation_centroid(2, 3) = -centroid(2);

	/// Compute the PCA
	pcl::PCA<PointT> pca;
	pca.setInputCloud(input_);
	Eigen::Vector3f eigenvalues = pca.getEigenValues();
	Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
	int axis = -1;
	if (eigenvalues(0) >= eigenvalues(1) && eigenvalues(0) >= eigenvalues(2)) {
		axis = 0;
	}
	else if (eigenvalues(1) >= eigenvalues(0) && eigenvalues(1) >= eigenvalues(2)) {
		axis = 1;
	}
	else {
		axis = 2;
	}

	/// Align the first PCA axis with the prealign axis
	Eigen::Vector3f vec_aux = eigenvectors.row(2);
	eigenvectors.row(2) = eigenvectors.row(axis);
	eigenvectors.row(axis) = vec_aux;

	float aux_ev = eigenvalues(2);
	eigenvalues(2) = eigenvalues(axis);
	eigenvalues(axis) = aux_ev;


	Eigen::Matrix<MatScalar, 4, 4> transformation_pca(Eigen::Matrix<MatScalar, 4, 4>::Identity());
	transformation_pca(0, 0) = eigenvectors(0, 0);
	transformation_pca(1, 0) = eigenvectors(0, 1);
	transformation_pca(2, 0) = eigenvectors(0, 2);

	transformation_pca(0, 1) = eigenvectors(1, 0);
	transformation_pca(1, 1) = eigenvectors(1, 1);
	transformation_pca(2, 1) = eigenvectors(1, 2);

	transformation_pca(0, 2) = eigenvectors(2, 0);
	transformation_pca(1, 2) = eigenvectors(2, 1);
	transformation_pca(2, 2) = eigenvectors(2, 2);

	transformation_prealign = transformation_pca * transformation_centroid;


	/// Set the variances
	eigenvalues /= static_cast<float> (input_->size());
	variances(0) = sqrt(eigenvalues(0));
	variances(1) = sqrt(eigenvalues(1));
	variances(2) = sqrt(eigenvalues(2));

}



////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename MatScalar> double
sq::SuperquadricFittingCeres<PointT, MatScalar>::fit(SuperquadricParameters<MatScalar> &parameters)
{
	//Eigen::Matrix<MatScalar, 4, 4> transformation_prealign(Eigen::Matrix<MatScalar, 4, 4>::Identity());
	//transformation_prealign = init_parameters_.transform;
	//input_prealigned_.reset(new Cloud());
	//pcl::transformPointCloud(*input_, *input_prealigned_, transformation_prealign);

	//CloudPtr inlier_cloud(new Cloud());
	//pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	//pcl::ExtractIndices<PointT> _extract;
	//for (size_t p_i = 0; p_i < input_prealigned_->size(); ++p_i)
	//{
	//	PointT px = input_prealigned_->points[p_i];
	//	if (sqrt(px.x * px.x + px.y * px.y) < 0.5)
	//		inlier_cloud->push_back(px);
	//}
	//
	///// Compute the PCA
	//pcl::PCA<PointT> pca;
	//pca.setInputCloud(inlier_cloud);
	//Eigen::Vector3f eigenvalues = pca.getEigenValues();

	//ceres::Problem problem;
	//const double doubleMax = std::numeric_limits<double>::max();
	//double xvec[11];
	//xvec[0] = init_parameters_.e1;
	//xvec[1] = init_parameters_.e2;
	//xvec[2] = sqrt(eigenvalues(0));
	//xvec[3] = sqrt(eigenvalues(1));
	//xvec[4] = sqrt(eigenvalues(2));
	//xvec[5] = xvec[6] = xvec[7] = xvec[8] = xvec[9] = xvec[10] = 0.;

	//std::cout <<input_prealigned_->size() << "," <<  xvec[0] << "," << xvec[1] << "," << xvec[2] << "," << xvec[3] << "," << xvec[4] << std::endl;
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viz;
	//viz.reset(new pcl::visualization::PCLVisualizer("parts Viewer"));
	////viz->setBackgroundColor(1.0f, 1.0f, 1.0f);
	//viz->addCoordinateSystem(0.1);
	//viz->addPointCloud(input_prealigned_, "cloud");
	//while (!viz->wasStopped())
	//	viz->spinOnce(1);
	//viz->close();

	Eigen::Matrix<MatScalar, 4, 4> transformation_prealign(Eigen::Matrix<MatScalar, 4, 4>::Identity());
	Eigen::Matrix<MatScalar, 3, 1> variances;
	variances(0) = variances(1) = variances(2) = static_cast <MatScalar> (1.);
	preAlign(transformation_prealign, variances);
	input_prealigned_.reset(new Cloud());
	pcl::transformPointCloud(*input_, *input_prealigned_, transformation_prealign);
	CloudPtr inlier_cloud(new Cloud());
	pcl::copyPointCloud(*input_prealigned_, *inlier_cloud);
	ceres::Problem problem;
	const double doubleMax = std::numeric_limits<double>::max();
	double xvec[11];
	xvec[0] = init_parameters_.e1;
	xvec[1] = init_parameters_.e2;
	xvec[2] = variances(0) * 3.;
	xvec[3] = variances(1) * 3.;
	xvec[4] = variances(2) * 3.;
	xvec[5] = xvec[6] = xvec[7] = xvec[8] = xvec[9] = xvec[10] = 0.;




	for (size_t p_i = 0; p_i < inlier_cloud->size(); ++p_i)
	{
		PointT &point = (*inlier_cloud)[p_i];
		ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<SuperquadricCostFunctor, 1, 11>(new SuperquadricCostFunctor(point));
		//    ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<SuperquadricCostFunctor, ceres::CENTRAL, 1, 11> (new SuperquadricCostFunctor (point));
		problem.AddResidualBlock(cost_function, NULL, xvec);
	}
	problem.SetParameterLowerBound(xvec, 0, 0.1);
	problem.SetParameterLowerBound(xvec, 1, 0.1);
	problem.SetParameterLowerBound(xvec, 2, 0.001);
	problem.SetParameterLowerBound(xvec, 3, 0.001);
	problem.SetParameterLowerBound(xvec, 4, 0.001);

	problem.SetParameterUpperBound(xvec, 0, 2.01);
	problem.SetParameterUpperBound(xvec, 1, 2.01);

	ceres::Solver::Options options;
	options.minimizer_type = ceres::TRUST_REGION;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	options.num_threads = 5;
	options.max_num_iterations = 1000;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	//Eigen::Matrix<MatScalar, 4, 4> &transformation = parameters.transform;
	//Eigen::Matrix<MatScalar, 4, 4> transformation;
	Eigen::Matrix<MatScalar, 4, 4> transformation(Eigen::Matrix<MatScalar, 4, 4>::Identity());
	//transformation.setZero();
	transformation(0, 3) = xvec[5];
	transformation(1, 3) = xvec[6];
	transformation(2, 3) = xvec[7];
	transformation(3, 3) = 1.;
	transformation.block(0, 0, 3, 3) = Eigen::AngleAxis<MatScalar>(xvec[8], Eigen::Matrix<MatScalar, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<MatScalar>(xvec[9], Eigen::Matrix<MatScalar, 3, 1>::UnitX()) *
		Eigen::AngleAxis<MatScalar>(xvec[10], Eigen::Matrix<MatScalar, 3, 1>::UnitZ()).matrix();


	//  clampParameters (xvec[0], xvec[1]);

	parameters.e1 = xvec[0];
	parameters.e2 = xvec[1];
	parameters.a = xvec[2];
	parameters.b = xvec[3];
	parameters.c = xvec[4];
	parameters.transform = Eigen::Matrix<MatScalar, 4, 4>(transformation) * transformation_prealign;

	//MatScalar final_error = computeSuperQuadricError<PointT, MatScalar>(input_, xvec[0], xvec[1], xvec[2], xvec[3], xvec[4], transformation);
	MatScalar final_error = 1.0;

	return (final_error);
}


////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename MatScalar>
template <typename T> bool
sq::SuperquadricFittingCeres<PointT, MatScalar>::SuperquadricCostFunctor::operator () (const T* const xvec, T* residual) const
{
	T e1 = xvec[0],
		e2 = xvec[1],
		a = xvec[2],
		b = xvec[3],
		c = xvec[4];
	Eigen::Matrix<T, 4, 4> transformation;
	transformation.setZero();
	transformation(0, 3) = xvec[5];
	transformation(1, 3) = xvec[6];
	transformation(2, 3) = xvec[7];
	transformation(3, 3) = T(1.);
	transformation.block(0, 0, 3, 3) = Eigen::AngleAxis<T>(xvec[8], Eigen::Matrix<T, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<T>(xvec[9], Eigen::Matrix<T, 3, 1>::UnitX()) *
		Eigen::AngleAxis<T>(xvec[10], Eigen::Matrix<T, 3, 1>::UnitZ()).matrix();

	Eigen::Matrix<T, 4, 1> xyz(T(point_.x), T(point_.y), T(point_.z), T(1.));
	Eigen::Matrix<T, 4, 1> xyz_tr = transformation * xyz;
	T op = Eigen::Matrix<T, 3, 1>(xyz_tr[0], xyz_tr[1], xyz_tr[2]).norm();

	residual[0] = op * superquadric_function_scale_weighting<T>(xyz_tr[0], xyz_tr[1], xyz_tr[2], e1, e2, a, b, c);
	//residual[0] = superquadric_function_scale_weighting<T>(xyz_tr[0], xyz_tr[1], xyz_tr[2], e1, e2, a, b, c);

	return (true);
}
