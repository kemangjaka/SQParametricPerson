#pragma once

#ifndef _SUPERQUADRICS_
#define _SUPERQUADRICS_

#include "../Main/Header.h"
#include "./modules/fit_superquadric_ceres.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>

#define SAMPLING_SIZE 70

class Superquadrics {
private:
	vector<sq::SuperquadricParameters<double>> sq_params;
	
	vector<sq::SuperquadricParameters<double>> ini_params;

	
	

public:
	Superquadrics();
	~Superquadrics();
	void setIniTransformParams(vector<PointT> joints3D);
	void computeSQ(vector<pcl::PointCloud<PointT>::Ptr> parted_cloud);
	vector<sq::SuperquadricParameters<double>> getParams();
	int GenerateSQUniformMesh(pcl::PolygonMesh::Ptr& mesh, sq::SuperquadricParameters<double> sq_params, PointT color);
};

#endif