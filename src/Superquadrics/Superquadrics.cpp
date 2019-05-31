#include "Superquadrics.h"
#include "../Utils/Utils.hpp"


double sgn(double a)
{
	if (a < 0.0)
		return -1.0;
	else if (a > 0.0)
		return 1.0;
	else
		return 0.0;
}


Superquadrics::Superquadrics()
{
	//set pre-defined initial parameters e.g. spherical for head.
	ini_params.resize(10);
	///Torso
	ini_params[0].e1 = 0.1;
	ini_params[0].e2 = 0.1;
	ini_params[1].e1 = 1.0;
	ini_params[1].e2 = 1.0;
	for (int i = 2; i < ini_params.size(); i++)
	{
		ini_params[i].e1 = 0.1;
		ini_params[i].e2 = 1.0;
	}
}

Superquadrics::~Superquadrics()
{

}


void Superquadrics::setIniTransformParams(vector<PointT> joints3D)
{
	for (int pdx = 1; pdx <= ini_params.size(); pdx++)
	{
		PointT normVec = { 0.0f, 0.0f, 0.0f };
		PointT centVec = {0.0f, 0.0f, 0.0f};
		switch (pdx) {
		case 1:
			normVec = subtract(joints3D[1], joints3D[20]);
			centVec = mean(joints3D[0], joints3D[1], joints3D[20]);
			break;
		case 2:
			centVec = joints3D[3];
			break;
		case 3:
			normVec = subtract(joints3D[4], joints3D[5]);
			centVec = mean(joints3D[4], joints3D[5]);
			break;
		case 4:
			normVec = subtract(joints3D[5], joints3D[6]);
			centVec = mean(joints3D[5], joints3D[6]);
			break;
		case 5:
			normVec = subtract(joints3D[8], joints3D[9]);
			centVec = mean(joints3D[8], joints3D[9]);
			break;
		case 6:
			normVec = subtract(joints3D[9], joints3D[10]);
			centVec = mean(joints3D[9], joints3D[10]);
			break;
		case 7:
			normVec = subtract(joints3D[12], joints3D[13]);
			centVec = mean(joints3D[12], joints3D[13]);
			break;
		case 8:
			normVec = subtract(joints3D[13], joints3D[14]);
			centVec = mean(joints3D[13], joints3D[14]);
			break;
		case 9:
			normVec = subtract(joints3D[16], joints3D[17]);
			centVec = mean(joints3D[16], joints3D[17]);
			break;
		case 10:
			normVec = subtract(joints3D[17], joints3D[18]);
			centVec = mean(joints3D[17], joints3D[18]);
			break;
		}
		double angle =  normVec.z / normVec.getVector3fMap().norm();

		Eigen::Matrix4d rot = Eigen::Matrix4d::Identity();
		Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
		Eigen::Matrix4d transform_1 = Eigen::Matrix4d::Identity();
		Eigen::Matrix4d transform_2 = Eigen::Matrix4d::Identity();
		if (normVec.x != 0.0)
		{
			
			transform_1(1, 1) = cos(angle);
			transform_1(1, 2) = sin(angle);
			transform_1(2, 1) = -sin(angle);
			transform_1(2, 2) = cos(angle);
			
			transform_2(1, 1) = 0.0;
			transform_2(1, 2) = 1.0;
			transform_2(2, 1) = -1.0;
			transform_2(2, 2) = 0.0;
			rot = transform_1 * transform_2;
		}
		if (centVec.z != 0.0f)
		{
			trans(0, 3) = -centVec.x;
			trans(1, 3) = -centVec.y;
			trans(2, 3) = -centVec.z;
		}
		ini_params[pdx - 1].transform = rot* trans;
	}


}

void Superquadrics::computeSQ(vector<pcl::PointCloud<PointT>::Ptr> parted_cloud)
{
	std::cout << "compute SQ! " << std::endl;
	sq_params.resize(ini_params.size());
#ifdef _OPENMP
omp_set_num_threads(10);
#pragma omp parallel for
#endif
	for (int l = 0; l < ini_params.size(); l++)
	{
		pcl::PointCloud<PointT>::Ptr cloud = parted_cloud[l];
		if (cloud->size() < 100)
			continue;
		// Create the filtering object
		pcl::StatisticalOutlierRemoval<PointT> sor;
		sor.setInputCloud(cloud);
		sor.setMeanK(50);
		sor.setStddevMulThresh(1.0);
		sor.filter(*cloud);
		sq::SuperquadricParameters<double> res_param;
		sq::SuperquadricFittingCeres<PointT, double> sq_fit;
		sq_fit.setInputCloud(cloud);
		sq_fit.setInitParameters(ini_params[l]);
		sq_fit.fit(res_param);
		sq_params[l] = res_param;
	}
}

vector<sq::SuperquadricParameters<double>> Superquadrics::getParams()
{
	return sq_params;
}

int Superquadrics::GenerateSQUniformMesh(pcl::PolygonMesh::Ptr& mesh, sq::SuperquadricParameters<double> sq_params,  PointT color)
{

	pcl::PointCloud<PointCT>::Ptr quadric(new pcl::PointCloud<PointCT>());
	double a1 = sq_params.a;
	double a2 = sq_params.b;
	double a3 = sq_params.c;
	double eps1 = sq_params.e1;
	double eps2 = sq_params.e2;
	
	Eigen::Matrix<double, 4, 4> transform = sq_params.transform.inverse();

	eps1 = abs(eps1);
	eps2 = abs(eps2);
	int phi_samples_ = SAMPLING_SIZE;
	int theta_samples_ = SAMPLING_SIZE;
	double theta_sample_rate = 2 * M_PI / (double)phi_samples_;
	double phi_sample_rate = M_PI / (double)theta_samples_;

	for (double phi = -M_PI / 2.0; phi < M_PI / 2.0; phi += phi_sample_rate)
		for (double theta = -M_PI; theta < M_PI; theta += theta_sample_rate)
		{
			double law = pow(pow(pow(abs(cos(phi) * cos(theta)), 2.0 / eps2) + pow(abs(cos(phi) * sin(theta)), 2.0 / eps2), eps2 / eps1) + pow(abs(sin(phi)), 2.0 / eps1), eps1 / -2.0);
			double x = sgn(cos(phi)) * a1 * law * abs(cos(phi)) * sgn(cos(theta)) * abs(cos(theta));
			double y = sgn(cos(phi)) * a2 * law * abs(cos(phi)) * sgn(sin(theta)) * abs(sin(theta));
			double z = sgn(sin(phi)) * a3 * law * abs(sin(phi));
			PointCT p;
			p.x = x;
			p.y = y;
			p.z = z;
			p.r = color.x;
			p.g = color.y;
			p.b = color.z;
			quadric->push_back(p);
		}

	for (int v = 1; v < phi_samples_; ++v)
	{
		for (int u = 1; u < theta_samples_; ++u)
		{
			pcl::Vertices polygon;
			polygon.vertices.push_back(v * phi_samples_ + u);
			polygon.vertices.push_back(v * phi_samples_ + u - 1);
			polygon.vertices.push_back((v - 1) * phi_samples_ + u);
			mesh->polygons.push_back(polygon);

			polygon.vertices.clear();
			polygon.vertices.push_back((v - 1) * phi_samples_ + u);
			polygon.vertices.push_back(v * phi_samples_ + u - 1);
			polygon.vertices.push_back((v - 1) * phi_samples_ + u - 1);
			mesh->polygons.push_back(polygon);
		}

		/// And connect the last column with the first one
		pcl::Vertices polygon;
		polygon.vertices.push_back(v * phi_samples_ + 0);
		polygon.vertices.push_back(v * phi_samples_ + phi_samples_ - 1);
		polygon.vertices.push_back((v - 1) * phi_samples_ + 0);
		mesh->polygons.push_back(polygon);

		polygon.vertices.clear();
		polygon.vertices.push_back((v - 1) * phi_samples_ + 0);
		polygon.vertices.push_back(v * phi_samples_ + phi_samples_ - 1);
		polygon.vertices.push_back((v - 1) * phi_samples_ + phi_samples_ - 1);
		mesh->polygons.push_back(polygon);
	}
	for (size_t i = 1; i < phi_samples_; ++i)
	{
		pcl::Vertices polygon;
		polygon.vertices.push_back(quadric->size() - 1);
		polygon.vertices.push_back(i - 1);
		polygon.vertices.push_back(i);

		mesh->polygons.push_back(polygon);
	}
	pcl::Vertices polygon;
	polygon.vertices.push_back(quadric->size() - 1);
	polygon.vertices.push_back(theta_samples_);
	polygon.vertices.push_back(0);
	mesh->polygons.push_back(polygon);

	transformPointCloud(*quadric, *quadric, transform);
	toPCLPointCloud2(*quadric, mesh->cloud);

	return 1;

}