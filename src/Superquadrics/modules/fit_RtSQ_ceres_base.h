#pragma once

#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#define MAX_DIST 0.75


namespace sq
{
template <typename T>
struct SuperquadricParameters;

template <typename PointT = pcl::PointXYZ, typename MatScalar = double>
class RtSQFittingCeresBase
{
  typedef pcl::PointCloud<PointT> Cloud;
  typedef typename Cloud::Ptr CloudPtr;
  typedef typename Cloud::ConstPtr CloudConstPtr;


public:
	RtSQFittingCeresBase(const SuperquadricParameters<MatScalar> parameters);


  void
  setInputCloud (const CloudConstPtr &cloud)
  { input_ = cloud; }

  SuperquadricParameters<MatScalar> prev_params;

  double
  fit (SuperquadricParameters<MatScalar> &parameters);


  struct SuperquadricCostFunctor
  {
    SuperquadricCostFunctor (const PointT &point, const SuperquadricParameters<MatScalar>& sq_params, const MatScalar _w)
    { point_ = point;
	_prevs = sq_params;
	weights = _w;
	}

    template <typename T> bool
    operator () (const T* const x, T* residual) const;

    PointT point_;
	SuperquadricParameters<MatScalar> _prevs;
	MatScalar weights;
  };


protected:
  CloudConstPtr input_;
  CloudPtr input_prealigned_;
  CloudPtr input_copy;

  

  bool pre_align_;
  int pre_align_axis_;

  double planeAngle;

};
}

#include "impl/fit_RtSQ_ceres_base.hpp"
