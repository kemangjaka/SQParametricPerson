#pragma once
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>

#include <ceres/ceres.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "../fit_RtSQ_ceres.h"
#include "../superquadric_formulas.h"


////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename MatScalar>
sq::RtSQFittingCeres<PointT, MatScalar>::RtSQFittingCeres(const SuperquadricParameters<MatScalar> parameters)
{
	prev_params.a = parameters.a;
	prev_params.b = parameters.b;
	prev_params.c = parameters.c;
	prev_params.e1 = parameters.e1;
	prev_params.e2 = parameters.e2;
	prev_params.transform = parameters.transform;
}


////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename MatScalar> double
sq::RtSQFittingCeres<PointT, MatScalar>::fit(SuperquadricParameters<MatScalar>& parameters)
{
	Eigen::Matrix<MatScalar, 4, 4> transformation_prealign(Eigen::Matrix<MatScalar, 4, 4>::Identity());
	transformation_prealign = prev_params.transform;

	input_prealigned_.reset(new Cloud());
	pcl::transformPointCloud(*input_, *input_prealigned_, transformation_prealign);
	//input_copy.reset(new Cloud());
	//copyPointCloud(*input_, *input_copy);
	
	ceres::Problem problem;
	const double doubleMax = std::numeric_limits<double>::max();
	double xvec[6];
	xvec[0] = xvec[1] = xvec[2] = xvec[3] = xvec[4] = xvec[5] = 0.;


	for (size_t p_i = 0; p_i < input_->size(); ++p_i)
	{
		//PointT &point = (*input_copy)[p_i];
		PointT &point = (*input_prealigned_)[p_i];
		PointT _p = (*input_prealigned_)[p_i];
		double dist = normalized_distance<double>(_p.x, _p.y, _p.z, prev_params.e1, prev_params.e2, prev_params.a, prev_params.b, prev_params.c);
		ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<SuperquadricCostFunctor, 1, 6>(new SuperquadricCostFunctor(point, prev_params, dist));
		problem.AddResidualBlock(cost_function, NULL, xvec);
	}

	ceres::Solver::Options options;
	options.minimizer_type = ceres::TRUST_REGION;
	//options.minimizer_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	options.num_threads = 5;
	options.max_num_iterations = 100;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);


	Eigen::Matrix<MatScalar, 4, 4> &transformation = parameters.transform;
	transformation.setZero();
	transformation(0, 3) = xvec[0];
	transformation(1, 3) = xvec[1];
	transformation(2, 3) = xvec[2];
	transformation(3, 3) = 1.;
	transformation.block(0, 0, 3, 3) = Eigen::AngleAxis<MatScalar>(xvec[3], Eigen::Matrix<MatScalar, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<MatScalar>(xvec[4], Eigen::Matrix<MatScalar, 3, 1>::UnitX()) *
		Eigen::AngleAxis<MatScalar>(xvec[5], Eigen::Matrix<MatScalar, 3, 1>::UnitZ()).matrix();
	
	//parameters.transform = Eigen::Matrix<MatScalar, 4, 4>(transformation) * transformation_prealign;

	MatScalar final_error = computeSuperQuadricError<PointT, MatScalar>(input_, prev_params.e1, prev_params.e2,
		prev_params.a, prev_params.b, prev_params.c, transformation);


	return (final_error);
}


////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename MatScalar>
template <typename T> bool
sq::RtSQFittingCeres<PointT, MatScalar>::SuperquadricCostFunctor::operator () (const T* const xvec, T* residual) const
{
	T   e1 = T(_prevs.e1),
		e2 = T(_prevs.e2),
		a  = T(_prevs.a),
		b  = T(_prevs.b),
		c  = T(_prevs.c);
	Eigen::Matrix<T, 4, 4> transformation;
	transformation.setZero();
	transformation(0, 3) = xvec[0];
	transformation(1, 3) = xvec[1];
	transformation(2, 3) = xvec[2];
	transformation(3, 3) = T(1.);
	transformation.block(0, 0, 3, 3) = Eigen::AngleAxis<T>(xvec[3], Eigen::Matrix<T, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<T>(xvec[4], Eigen::Matrix<T, 3, 1>::UnitX()) *
		Eigen::AngleAxis<T>(xvec[5], Eigen::Matrix<T, 3, 1>::UnitZ()).matrix();

	Eigen::Matrix<T, 4, 1> xyz(T(point_.x), T(point_.y), T(point_.z), T(1.));
	//Eigen::Matrix<T, 4, 1> xyz_tr = _prevs.transform.cast<T>() * transformation * xyz;
	Eigen::Matrix<T, 4, 1> xyz_tr = transformation * xyz;
	T point_weight;
	//T dist = normalized_distance<T>(T(xyz_tr[0]), T(xyz_tr[1]), T(xyz_tr[2]), e1, e2, a, b, c);
	T dist = T(weights);
	if (dist > T(MAX_DIST))
	{
		//point_weight = 1.0;
		point_weight = dist - T(MAX_DIST);
		//point_weight = weights;
	}
	else
		point_weight = T(0.0);
	residual[0] = point_weight * superquadric_function<T>(xyz_tr[0], xyz_tr[1], xyz_tr[2], e1, e2, a, b, c);
	return (true);
}
