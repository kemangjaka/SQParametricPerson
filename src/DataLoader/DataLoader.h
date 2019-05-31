#pragma once

#include "../Main/Header.h"
#include <sstream>
#include <fstream>
#include <iostream>

class DataLoader {
private:
	string root_dir;
	string color_dir;
	string depth_dir;
	string colorDepth_dir;
	string joint_dir;
	string pcd_dir;
	string coord_dir;
	int frame_idx;

	int frame_max_idx;

public:
	DataLoader(string root_dir);
	~DataLoader();

	bool nextDataAvailable();

	cv::Mat getNextColorImage();
	cv::Mat getNextDepthImage();
	cv::Mat getNextColorDepthImage();
	cv::Mat getNextCoordMapper();
	pcl::PointCloud<PointCT>::Ptr getNextPointCloud();
	int getNext2D3DJoints(vector<vector<PointT>>& joints3D, vector<vector<cv::Point2i>>& joints2D);
	void readyForNextFrame();
};