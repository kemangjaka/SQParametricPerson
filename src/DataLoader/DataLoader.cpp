#include "DataLoader.h"
#include <boost/filesystem.hpp>
int count_files(std::string directory, std::string ext)
{
	namespace fs = boost::filesystem;
	fs::path Path(directory);
	int Nb_ext = 0;
	fs::directory_iterator end_iter; // Default constructor for an iterator is the end iterator

	for (fs::directory_iterator iter(Path); iter != end_iter; ++iter)
		if (iter->path().extension() == ext)
			++Nb_ext;

	return Nb_ext;
}


DataLoader::DataLoader(string _root) 
{
	root_dir = _root;
	color_dir = root_dir + "/colorImages/";
	depth_dir = root_dir + "/depthImages/";
	colorDepth_dir = root_dir + "/colorDepthImages/";
	joint_dir = root_dir + "/joints/";
	pcd_dir = root_dir + "/pointclouds/";
	coord_dir = root_dir + "/correspond/";

	frame_max_idx = count_files(color_dir.c_str(), ".jpg");
	std::cout << frame_max_idx << " files found! " << std::endl;
	frame_idx = 0;
}

DataLoader::~DataLoader()
{

}

bool DataLoader::nextDataAvailable()
{
	return frame_idx < frame_max_idx;
}

cv::Mat DataLoader::getNextColorImage()
{
	std::cout << color_dir + to_string(frame_idx) + ".jpg" << std::endl;
	cv::Mat img = cv::imread(color_dir + to_string(frame_idx) + ".jpg");
	return img;
}

cv::Mat DataLoader::getNextDepthImage()
{
	cv::Mat img = cv::imread(depth_dir + to_string(frame_idx) + ".jpg", -1);
	return img;
}


cv::Mat DataLoader::getNextColorDepthImage()
{
	cv::Mat img = cv::imread(colorDepth_dir + to_string(frame_idx) + ".jpg");
	return img;
}

pcl::PointCloud<PointCT>::Ptr DataLoader::getNextPointCloud()
{
	pcl::PointCloud<PointCT>::Ptr cloud(new pcl::PointCloud<PointCT>());
	pcl::io::loadPCDFile(pcd_dir + to_string(frame_idx) + ".pcd", *cloud);
	return cloud;
}

cv::Mat DataLoader::getNextCoordMapper()
{
	cv::FileStorage fs(coord_dir + to_string(frame_idx) + ".xml", cv::FileStorage::READ);
	cv::Mat corresMat;
	fs["correspondMatrix"] >> corresMat;
	return corresMat;
}

///memory should be allocated beforehand
int DataLoader::getNext2D3DJoints(vector<vector<PointT>>& joints3D, vector<vector<cv::Point2i>>& joints2D) 
{
	ifstream ifs(joint_dir + to_string(frame_idx) + ".csv");
	string str;
	int line_idx = 0;
	int personNum = 0;
	while (getline(ifs, str))
	{
		int personIdx = line_idx / 25;
		int jointIdx = line_idx % 25;
		string token;
		istringstream stream(str);
		int count = 0;
		while (getline(stream, token, ','))
		{
			if (count == 2)
				joints2D[personIdx][jointIdx].x = stoi(token);
			else if (count == 3)
				joints2D[personIdx][jointIdx].y = stoi(token);
			else if (count == 4)
				joints3D[personIdx][jointIdx].x = stof(token);
			else if (count == 5)
				joints3D[personIdx][jointIdx].y = stof(token);
			else if (count == 6)
				joints3D[personIdx][jointIdx].z = stof(token);
			count++;
		}
		line_idx++;
		if (jointIdx == 0)
			if (joints2D[personIdx][jointIdx].x != 0)
				personNum++;
	}
	return personNum;
}

void DataLoader::readyForNextFrame()
{
	frame_idx++;
}