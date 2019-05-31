#include "Kinectv2.h"

template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL) {
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}


Kinectv2::Kinectv2()
{
	hResult = S_OK;

	hResult = GetDefaultKinectSensor(&pSensor);
	if (FAILED(hResult)) {
		Error_check("GetDefaultKinectSensor");
	}

	hResult = pSensor->Open();
	if (FAILED(hResult)) {
		Error_check("Open()");
	}
	colorWidth = 0;
	colorHeight = 0;

	pColorFrame = nullptr;
	pDepthFrame = nullptr;
	bodyFrame = nullptr;
}

Kinectv2::~Kinectv2()
{
	SafeRelease(pColorSource);
	SafeRelease(pDepthSource);
	SafeRelease(pColorReader);
	SafeRelease(pDepthReader);
	SafeRelease(pColorDescription);
	SafeRelease(pDepthDescription);
	SafeRelease(pCoordinateMapper);
	if (pSensor) {
		pSensor->Close();
	}
	SafeRelease(pSensor);

}

int Kinectv2::Error_check(std::string s)
{
	std::cout << "Error:" + s << std::endl;
	return -1;
}

void Kinectv2::UseColorImage()
{
	hResult = pSensor->get_ColorFrameSource(&pColorSource);
	if (FAILED(hResult)) {
		Error_check("get_ColorFrameSource");
	}

	hResult = pColorSource->OpenReader(&pColorReader);
	if (FAILED(hResult)) {
		Error_check("OpenReader");
	}
	hResult = pColorSource->get_FrameDescription(&pColorDescription);
	if (FAILED(hResult)) {
		Error_check("get_FrameDescription");
	}

	pColorDescription->get_Width(&colorWidth);
	pColorDescription->get_Height(&colorHeight);
	colorBuffer.resize(colorHeight * colorWidth);
}

void Kinectv2::UseDepthImage()
{
	hResult = pSensor->get_DepthFrameSource(&pDepthSource);
	if (FAILED(hResult)) {
		Error_check("get_DepthFrameSource");
	}
	hResult = pDepthSource->OpenReader(&pDepthReader);
	if (FAILED(hResult)) {
		Error_check("OpenReader");
	}
	hResult = pDepthSource->get_FrameDescription(&pDepthDescription);
	if (FAILED(hResult)) {
		Error_check("get_FrameDescription");
	}


	depthBuffer.resize(DEPTH_HEIGHT * DEPTH_WIDTH);

}


void Kinectv2::UseCoordinate()
{
	hResult = pSensor->get_CoordinateMapper(&pCoordinateMapper);
	if (FAILED(hResult)) {
		Error_check("get_CoordinateMapper");
	}

}

void Kinectv2::UseBodyEstimation()
{
	hResult = pSensor->get_BodyFrameSource(&bodyFrameSource);
	if (FAILED(hResult)) {
		Error_check("get_BodyFrameSource");
	}
	hResult = bodyFrameSource->OpenReader(&bodyFrameReader);
	if (FAILED(hResult)) {
		Error_check("OpenReader");
	}
}

void Kinectv2::UseBodySegmentation()
{
	hResult = pSensor->get_BodyIndexFrameSource(&pBodyIndexSource);
	if (FAILED(hResult)) {
		std::cerr << "Error : IKinectSensor::get_BodyIndexFrameSource()" << std::endl;
	}
	hResult = pBodyIndexSource->OpenReader(&pBodyIndexReader);
	if (FAILED(hResult)) {
		std::cerr << "Error : IBodyIndexFrameSource::OpenReader()" << std::endl;
	}
}

cv::Mat Kinectv2::getColorImage()
{
	unsigned int bufferSize = colorWidth * colorHeight * 4 * sizeof(unsigned char);
	bufferMat = cv::Mat(colorHeight, colorWidth, CV_8UC4);
	colorMat = cv::Mat(colorHeight / 2, colorWidth / 2, CV_8UC4);

	//colorMat = cv::Mat(colorHeight ,colorWidth,CV_8UC4);
	hResult = pColorReader->AcquireLatestFrame(&pColorFrame);
	if (SUCCEEDED(hResult)) {
		hResult = pColorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD),
			reinterpret_cast<BYTE*>(&colorBuffer[0]), ColorImageFormat::ColorImageFormat_Bgra);
		hResult = pColorFrame->CopyConvertedFrameDataToArray(bufferSize, reinterpret_cast<BYTE*>
			(bufferMat.data), ColorImageFormat_Bgra);
		if (FAILED(hResult)) {
			std::cerr << "Error:IColorFrame::CopyConvertedFrameDataToArray()" << std::endl;
		}
		cv::resize(bufferMat, colorMat, cv::Size(), 0.5, 0.5);
	}
	cv::flip(colorMat, colorMat, 1);
	SafeRelease(pColorFrame);
	//cvtColor(colorMat, colorMat, CV_RGBA2BGR);
	return colorMat;
}


cv::Mat Kinectv2::getDepthImage()
{

	unsigned int dBufferSize = DEPTH_WIDTH * DEPTH_HEIGHT * sizeof(unsigned char);
	dBufferMat = cv::Mat(DEPTH_HEIGHT, DEPTH_WIDTH, CV_16UC1);
	depthMat = cv::Mat(DEPTH_HEIGHT, DEPTH_WIDTH, CV_16UC1);

	hResult = pDepthReader->AcquireLatestFrame(&pDepthFrame);
	if (SUCCEEDED(hResult)) {
		hResult = pDepthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0]);

		hResult = pDepthFrame->AccessUnderlyingBuffer(&dBufferSize, reinterpret_cast<UINT16**>(&dBufferMat.data));
		if (SUCCEEDED(hResult)) {
			dBufferMat.convertTo(depthMat, CV_8U, -255.0f / 8000.0f, 255.0f);
		}
		else if (FAILED(hResult)) {
			std::cerr << "Error:IDepthFrame::CopyConvertedFrameDataToArray()" << std::endl;
		}
	}
	SafeRelease(pDepthFrame);
	flip(depthMat, depthMat, 1);

	return depthMat;
}

int Kinectv2::getBodyPoints(vector<vector<cv::Point2i>>& joint2d, vector<vector<PointT>>& joint3d)
{
	IBody* bodies[6] = { 0 };
	if (bodyFrameReader == nullptr)
		return 0;
	hResult = bodyFrameReader->AcquireLatestFrame(&bodyFrame);
	int index = 0;
	if (SUCCEEDED(hResult))
	{

		bodyFrame->GetAndRefreshBodyData(6, bodies);
		joint2d.resize(6);
		joint3d.resize(6);
		for (int i = 0; i < 6; i++)
		{
			joint2d[i].resize(JointType::JointType_Count);
			joint3d[i].resize(JointType::JointType_Count);
			fill(joint2d[i].begin(), joint2d[i].end(), cv::Point2i(0, 0));
			fill(joint3d[i].begin(), joint3d[i].end(), PointT(0.0, 0.0, 0.0));
		}



		
		for (auto body : bodies)
		{
			if (body == nullptr)
				continue;
			Joint joints[JointType::JointType_Count];
			body->GetJoints(JointType::JointType_Count, joints);

			for (auto joint : joints)
			{
				int label = joint.JointType;
				PointT p;
				p.x = joint.Position.X;
				p.y = joint.Position.Y;
				p.z = joint.Position.Z;
				//just store one person
				if (p.x == 0.0 && p.y == 0.0 && p.z == 0.0)
					break;
				joint3d[index][label] = p;
				cv::Point2i cp;
				//ColorSpacePoint point;
				DepthSpacePoint point;
				pCoordinateMapper->MapCameraPointToDepthSpace(joint.Position, &point);
				//pCoordinateMapper->MapCameraPointToColorSpace(joint.Position, &point);
				cp.x = DEPTH_WIDTH - point.X;
				cp.y = point.Y;
				joint2d[index][label] = cp;
			}
			index++;
		}

	}
	SafeRelease(bodyFrame);

	return index;
}

cv::Mat Kinectv2::MapLabelImageToDepth(cv::Mat label)
{
	cv::resize(label, label, cv::Size(), 2.0, 2.0);
	cv::flip(label, label, 1);
	cv::Mat t_label = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGHT), CV_8UC1);
	for(int y = 0;y < DEPTH_HEIGHT;y++)
		for (int x = 0; x < DEPTH_WIDTH; x++)
		{
			DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
			UINT16 depth = depthBuffer[y * DEPTH_WIDTH + x];
			unsigned char real_depth = depthMat.at<unsigned char>(y, DEPTH_WIDTH - x);
			if (real_depth == 0 || real_depth == 255)
				continue;
			ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
			pCoordinateMapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
			int colorX = static_cast<int>(colorSpacePoint.X);
			int colorY = static_cast<int>(colorSpacePoint.Y);
			//int colorX = static_cast<int>(floor(colorSpacePoint.X + 0.5f));
			//int colorY = static_cast<int>(floor(colorSpacePoint.Y + 0.5f));
			if (0 <= colorX && colorX < colorWidth && 0 <= colorY && colorY < colorHeight)
			{
				unsigned char l = label.at<unsigned char>(colorY, colorX);
				t_label.at<unsigned char>(y, x) = l;
			}
		}
	cv::flip(t_label, t_label, 1);
	return t_label;
}

cv::Mat Kinectv2::getColorDepthImage()
{
	std::vector<ColorSpacePoint> colorSpacePoints(depthBuffer.size());
	hResult = pCoordinateMapper->MapDepthFrameToColorSpace(depthBuffer.size(), &depthBuffer[0], colorSpacePoints.size(), &colorSpacePoints[0]);

	cv::Mat colorDepthImage(DEPTH_HEIGHT, DEPTH_WIDTH, CV_8UC4);
	if (SUCCEEDED(hResult))
	{
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < DEPTH_WIDTH * DEPTH_HEIGHT; i++)
		{
			int x = (int)colorSpacePoints[i].X;
			int y = (int)colorSpacePoints[i].Y;
			int ColorBytesPerPixel = 4;
			int srcIndex = y * colorWidth + x;
			//	int srcIndex = ((y * colorWidth / 2 + x)) * ColorBytesPerPixel;
			int destIndex = i * ColorBytesPerPixel;

			if (((0 <= x) && (x < colorWidth) && (0 <= y) && (y < colorHeight)))
			{
				colorDepthImage.data[destIndex + 0] = colorBuffer[srcIndex].rgbBlue;
				colorDepthImage.data[destIndex + 1] = colorBuffer[srcIndex].rgbGreen;
				colorDepthImage.data[destIndex + 2] = colorBuffer[srcIndex].rgbRed;
			}
			else {
				colorDepthImage.data[destIndex + 0] = 255;
				colorDepthImage.data[destIndex + 1] = 255;
				colorDepthImage.data[destIndex + 2] = 255;

			}
		}
		cv::flip(colorDepthImage, colorDepthImage, 1);
	}
	//cvtColor(colorDepthImage, colorDepthImage, CV_RGBA2BGR);
	return colorDepthImage;



}


void Kinectv2::Activatev2()
{
	UseColorImage();
	UseDepthImage();
	UseBodyEstimation();
	UseCoordinate();


}

cv::Mat Kinectv2::getBodySegmImage()
{
	cv::Mat bodySegmImg = cv::Mat::zeros(DEPTH_HEIGHT, DEPTH_WIDTH, CV_8U);
	std::cout << bodySegmImg.size() << std::endl;
	if (pBodyIndexReader == nullptr)
		return bodySegmImg;
	IBodyIndexFrame* pBodyIndexFrame = nullptr;
	std::cout << hResult << std::endl;
	hResult = pBodyIndexReader->AcquireLatestFrame(&pBodyIndexFrame);
	
	if (SUCCEEDED(hResult)) {
		unsigned int bufferSize = 0;
		unsigned char* buffer = nullptr;
		hResult = pBodyIndexFrame->AccessUnderlyingBuffer(&bufferSize, &buffer);
		std::cout << hResult << std::endl;
		if (SUCCEEDED(hResult)) {
			for (int y = 0; y < DEPTH_HEIGHT; y++) {
				for (int x = 0; x < DEPTH_WIDTH; x++) {
					unsigned int index = y * DEPTH_WIDTH + x;
					std::cout << static_cast<int>(buffer[index]) << std::endl;
					if (buffer[index] != 0xff)
						bodySegmImg.at<unsigned char>(y, x) = buffer[index];
				}
			}
		}
	}
	SafeRelease(pBodyIndexFrame);
	cv::flip(bodySegmImg, bodySegmImg, 1);
	return bodySegmImg;
}

pcl::PointCloud<PointT>::Ptr Kinectv2::getPointCloud()
{
	pcl::PointCloud<PointT>::Ptr points(new pcl::PointCloud<PointT>);
	// PointCloudのサイズ指定
	points->width = DEPTH_WIDTH;
	points->height = DEPTH_HEIGHT;
	points->resize(points->width * points->height);
	for (int y = 0; y < points->height; y++)
		for (int x = 0; x < points->width; x++)
		{
			DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
			UINT16 depth = depthBuffer[y * DEPTH_WIDTH + x];
			CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
			pCoordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
			if (cameraSpacePoint.X == std::numeric_limits<double>::infinity() || cameraSpacePoint.X == -1 * std::numeric_limits<double>::infinity())
				cameraSpacePoint.X = cameraSpacePoint.Y = cameraSpacePoint.Z = 0.0;
			points->at(x, y).x = cameraSpacePoint.X;
			points->at(x, y).y = cameraSpacePoint.Y;
			points->at(x, y).z = cameraSpacePoint.Z;

		}
	return points;

}

pcl::PointCloud<PointCT>::Ptr Kinectv2::getColorPointCloud()
{
	pcl::PointCloud<PointCT>::Ptr points(new pcl::PointCloud<PointCT>);
	// PointCloudのサイズ指定
	points->width = DEPTH_WIDTH;
	points->height = DEPTH_HEIGHT;
	points->resize(points->width * points->height);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < points->height; y++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < points->width; x++)
		{
			DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
			UINT16 depth = depthBuffer[y * DEPTH_WIDTH + x];
			CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
			pCoordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
			if (cameraSpacePoint.X == std::numeric_limits<double>::infinity() || cameraSpacePoint.X == -1 * std::numeric_limits<double>::infinity())
				cameraSpacePoint.X = cameraSpacePoint.Y = cameraSpacePoint.Z = 0.0;

			ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
			pCoordinateMapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
			int colorX = static_cast<int>(floor(colorSpacePoint.X + 0.5f));
			int colorY = static_cast<int>(floor(colorSpacePoint.Y + 0.5f));
			if (0 <= colorX && colorX < colorWidth && 0 <= colorY && colorY < colorHeight)
			{
				RGBQUAD color = colorBuffer[colorY * colorWidth + colorX];
				points->at(x, y).b = color.rgbBlue;
				points->at(x, y).g = color.rgbGreen;
				points->at(x, y).r = color.rgbRed;
			}
			points->at(x, y).x = cameraSpacePoint.X;
			points->at(x, y).y = cameraSpacePoint.Y;
			points->at(x, y).z = cameraSpacePoint.Z;

		}
	return points;

}

pcl::PointCloud<PointCT>::Ptr Kinectv2::getRangedColorPointCloud(float max_dist)
{
	pcl::PointCloud<PointCT>::Ptr points(new pcl::PointCloud<PointCT>);
	// PointCloudのサイズ指定
	points->width = DEPTH_WIDTH;
	points->height = DEPTH_HEIGHT;
	points->resize(points->width * points->height);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < points->height; y++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < points->width; x++)
		{
			DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
			UINT16 depth = depthBuffer[y * DEPTH_WIDTH + x];
			CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
			pCoordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
			if (cameraSpacePoint.X == std::numeric_limits<double>::infinity() || cameraSpacePoint.X == -1 * std::numeric_limits<double>::infinity())
				cameraSpacePoint.X = cameraSpacePoint.Y = cameraSpacePoint.Z = 0.0;

			if(sqrt(cameraSpacePoint.X * cameraSpacePoint.X + cameraSpacePoint.Y * cameraSpacePoint.Y + cameraSpacePoint.Z * cameraSpacePoint.Z) > max_dist)
				cameraSpacePoint.X = cameraSpacePoint.Y = cameraSpacePoint.Z = 0.0;
			else
			{

				ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
				pCoordinateMapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
				int colorX = static_cast<int>(floor(colorSpacePoint.X + 0.5f));
				int colorY = static_cast<int>(floor(colorSpacePoint.Y + 0.5f));
				if (0 <= colorX && colorX < colorWidth && 0 <= colorY && colorY < colorHeight)
				{
					RGBQUAD color = colorBuffer[colorY * colorWidth + colorX];
					points->at(x, y).b = color.rgbBlue;
					points->at(x, y).g = color.rgbGreen;
					points->at(x, y).r = color.rgbRed;
				}
			}
			points->at(x, y).x = cameraSpacePoint.X;
			points->at(x, y).y = cameraSpacePoint.Y;
			points->at(x, y).z = cameraSpacePoint.Z;

		}
	return points;

}


std::array<double, 3> Kinectv2::getPointData(int x, int y, int k)
{
	std::array<double, 3> point;
	DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
	UINT16 depth = depths[k][y * DEPTH_WIDTH + x];
	std::vector<CameraSpacePoint> cameraPoints;
	// Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
	CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };

	pCoordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
	point[0] = cameraSpacePoint.X;
	point[1] = cameraSpacePoint.Y;
	point[2] = cameraSpacePoint.Z;
	if (point[0] == std::numeric_limits<double>::infinity() || point[0] == -1 * std::numeric_limits<double>::infinity())
		point[0] = point[1] = point[2] = 0.0;
	return point;
}




