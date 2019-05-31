#pragma once


#include "Header.h"
#include "../Superquadrics/Superquadrics.h"


#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>


/*
Body Definition
1: Torso (Kinect index : 0, 1, 20)
2: Head (Kinect Index : 3)
3: Left upper arm (Kinect Index : 4,5)
4: Left lower arm (Kinect Index : 5,6)
5: Right upper arm (Kinect Index : 8,9)
6: Right lower arm (Kinect Index : 9,10)
7: Left Thigh (Kinect Index : 12, 13)
8: Left Leg (Kinect Index : 13, 14)
9: Right Thigh ( Kinect Index: 16, 17)
10: Right Leg (Kinect Index : 17, 18)


PASCAL-person-part
1: Head
2: Torso
3: Upper arm
4: Lower arm
5: Thigh
6: Leg
*/

#define V2_WIDTH 512
#define V2_HEIGHT 424
#define MAX_LABEL 1000
#define JOINTS_NUM 10

class SemSeg;
class Superquadrics;
class Kinectv2;
class DataLoader;

class MainEngine {
private:
	vector<cv::Vec3b> random_colors;
	Kinectv2* kinectv2;
	SemSeg* segmEngine;
	Superquadrics* quadEngine;
	DataLoader* loadEngine;
	int width;
	int height;

	cv::Mat colorImage;
	cv::Mat depthImage;
	cv::Mat colorDepthImage;
	cv::Mat color2Depth;

	vector<pcl::PointCloud<PointT>::Ptr> partedCloud;

	vector<vector<PointT>> joints3D;
	vector<vector<cv::Point2i>> joints2D;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> pcd_viz;

	vector<sq::SuperquadricParameters<double>> sq_params;

	pcl::PointCloud<PointCT>::Ptr JumpEdgeFilter(pcl::PointCloud<PointCT>::Ptr cloud);
	void CutZFilter(pcl::PointCloud<PointCT>::Ptr& cloud);
	cv::Mat PartsSegmentation(vector<vector<cv::Point2i>> joints2D, cv::Mat labelMap, int& activePersonIndex);
	void renderSQ();
	void jointVisualization(pcl::PointCloud<PointCT>::Ptr filtered);
	cv::Mat warpLabelImg(cv::Mat labelImg, cv::Mat warp);
public:
	MainEngine(int act_type);
	~MainEngine();
	void Activate();
	void ActivateLoadedData();
};