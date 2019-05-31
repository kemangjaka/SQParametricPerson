#ifndef _KINECTV2_
#define _KINECTV2_
#include <Kinect.h>
#include <Windows.h>

#include "../Main/Header.h"

#define DEPTH_HEIGHT 424
#define DEPTH_WIDTH 512

class Kinectv2
{
private:
	IKinectSensor* pSensor;
	HRESULT hResult;
	ICoordinateMapper* pCoordinateMapper;
	IColorFrameSource* pColorSource;
	IColorFrameReader* pColorReader;
	IDepthFrameSource* pDepthSource;
	IDepthFrameReader* pDepthReader;
	IBodyFrameSource* bodyFrameSource;
	IBodyFrameReader* bodyFrameReader;
	IFrameDescription* pColorDescription;
	IFrameDescription* pDepthDescription;
	IBodyIndexFrameSource* pBodyIndexSource;
	IBodyIndexFrameReader* pBodyIndexReader;
	IColorFrame* pColorFrame;
	IDepthFrame* pDepthFrame;
	IBodyFrame* bodyFrame;
	int colorWidth;
	int colorHeight;
	
	std::vector<RGBQUAD> colorBuffer;
	std::vector<UINT16> depthBuffer;
	std::vector<UINT16> rawDepth;
	std::vector<std::vector<UINT16>> depths;
	std::vector<UINT16> afdepthBuffer;
	cv::Mat colorMat;
	cv::Mat bufferMat;
	cv::Mat dBufferMat;
	cv::Mat depthMat;
	int time;
	UINT16 depth;

	std::vector<ColorSpacePoint> colorSpace;

	void UseColorImage();
	void UseDepthImage();
	void UseCoordinate();
	void UseBodyEstimation();
	void UseBodySegmentation();

public:
	Kinectv2();
	~Kinectv2();
	void Activatev2(); //Ready to Use Kinect v2 open the stream of Depth and Color
	int Error_check(std::string s);

	cv::Mat getColorImage();
	cv::Mat getDepthImage();
	int getBodyPoints(vector<vector<cv::Point2i>>& joint2d, vector<vector<PointT>>& joint3d);
	pcl::PointCloud<PointT>::Ptr getPointCloud();
	pcl::PointCloud<PointCT>::Ptr getColorPointCloud();
	pcl::PointCloud<PointCT>::Ptr getRangedColorPointCloud(float max_dist);
	cv::Mat MapLabelImageToDepth(cv::Mat label);
	std::array<double, 3> getPointData(int x, int y, int k);
	cv::Mat getColorDepthImage();
	cv::Mat getBodySegmImage();
};

#endif