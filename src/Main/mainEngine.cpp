#include "mainEngine.h"
#include "../Utils/Utils.hpp"
#include "../Kinectv2/Kinectv2.h"
#include "../SemanticSegmentation/SemanticSegmentation.h"
#include "../DataLoader/DataLoader.h"

MainEngine::MainEngine(int act_type)
{


	//visualEngine = new Visualization();

	pcd_viz.reset(new pcl::visualization::PCLVisualizer("3D Viewer"));
	pcd_viz->setBackgroundColor(1.0, 1.0, 1.0);
	pcd_viz->addCoordinateSystem(0.1);
	pcd_viz->loadCameraParameters("cam.cam");


	random_colors.resize(MAX_LABEL);
	for (int i = 0; i < random_colors.size(); i++)
		random_colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
	random_colors[0] = cv::Vec3b(0, 0, 0);

	width = V2_WIDTH;
	height = V2_HEIGHT;

	if (act_type == 0)
	{
		kinectv2 = new Kinectv2();
		kinectv2->Activatev2();
	}
	else
	{
		loadEngine = new DataLoader("D:\\ProjectsD\\ISMAR2019\\dataset\\data_1559090190");
	}
	segmEngine = new SemSeg();
	while (!segmEngine->isReady())
		continue;
	quadEngine = new Superquadrics();


	partedCloud.resize(JOINTS_NUM);
	for (int l = 0; l < partedCloud.size(); l++)
		partedCloud[l].reset(new pcl::PointCloud<PointT>());
	//

	joints2D.resize(6);
	joints3D.resize(6);
	for (int i = 0; i < 6; i++)
	{
		joints2D[i].resize(25);
		joints3D[i].resize(25);
		fill(joints2D[i].begin(), joints2D[i].end(), cv::Point2i(0, 0));
		fill(joints3D[i].begin(), joints3D[i].end(), PointT(0.0, 0.0, 0.0));
	}

}

MainEngine::~MainEngine()
{

}

cv::Mat MainEngine::PartsSegmentation(vector<vector<cv::Point2i>> joints2D, cv::Mat labelMap, int& activePersonIndex)
{
	cv::Mat partIdImg = cv::Mat::zeros(colorDepthImage.size(), CV_8U);
	for (int pdx = 0; pdx < 6; pdx++)
	{
		vector<cv::Point2i> person_joints = joints2D[pdx];

		int sum = 0;
		for (int i = 0; i < person_joints.size(); i++)
			sum += person_joints[i].x;
		if (sum == 0)
			continue;
		activePersonIndex = pdx;
		for (int jdx = 1; jdx < 7; jdx++)
		{
			std::vector<cv::Point2i> corr_joints2d;
			cv::Mat corr_semParts = (labelMap == jdx);
			switch (jdx) {
			case 1: ///torso
			case 2: ///head
				break;
			case 3: ///upper arm (left, right)
				corr_joints2d.push_back(person_joints[4]);
				corr_joints2d.push_back(person_joints[5]);
				corr_joints2d.push_back(person_joints[8]);
				corr_joints2d.push_back(person_joints[9]);
				break;
			case 4: ///lower arm (left, right)
				corr_joints2d.push_back(person_joints[5]);
				corr_joints2d.push_back(person_joints[6]);
				corr_joints2d.push_back(person_joints[9]);
				corr_joints2d.push_back(person_joints[10]);
				break;
			case 5: /// thigh (left, right)
				corr_joints2d.push_back(person_joints[12]);
				corr_joints2d.push_back(person_joints[13]);
				corr_joints2d.push_back(person_joints[16]);
				corr_joints2d.push_back(person_joints[17]);
				break;
			case 6: ///left leg
				corr_joints2d.push_back(person_joints[13]);
				corr_joints2d.push_back(person_joints[14]);
				corr_joints2d.push_back(person_joints[17]);
				corr_joints2d.push_back(person_joints[18]);
			}

			cv::morphologyEx(corr_semParts, corr_semParts, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
			cv::morphologyEx(corr_semParts, corr_semParts, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);

			switch (jdx)
			{
			case 1:
				partIdImg.setTo(2, corr_semParts);
				break;
			case 2:
				partIdImg.setTo(1, corr_semParts);
				break;
			default:
				for (int y = 0; y < corr_semParts.rows; y++)
					for (int x = 0; x < corr_semParts.cols; x++)
					{
						if (corr_semParts.at<unsigned char>(y, x) == 0)
							continue;
						float min_dist = 0.0f;
						int min_idx = -1;
						for (int cj = 0; cj < corr_joints2d.size(); cj++)
						{
							float dist = sqrt(pow(abs(x - corr_joints2d[cj].x), 2.0) + pow(abs(y - corr_joints2d[cj].y), 2.0));
							if (min_dist > dist || cj == 0)
							{
								min_dist = dist;
								min_idx = cj;
							}
						}
						//std::cout << min_dist << std::endl;
						if (min_idx == -1)
							continue;
						int instance_jidx = -1;
						if (min_idx < 2)
						{
							switch (jdx) {
							case 3:
								instance_jidx = 3;
								break;
							case 4:
								instance_jidx = 4;
								break;
							case 5:
								instance_jidx = 7;
								break;
							case 6:
								instance_jidx = 8;
								break;
							}
						}
						else {
							switch (jdx) {
							case 3:
								instance_jidx = 5;
								break;
							case 4:
								instance_jidx = 6;
								break;
							case 5:
								instance_jidx = 9;
								break;
							case 6:
								instance_jidx = 10;
								break;
							}
						}

						partIdImg.at<unsigned char>(y, x) = instance_jidx;
					}
			}
		}
	}
	return partIdImg;
}


pcl::PointCloud<PointCT>::Ptr MainEngine::JumpEdgeFilter(pcl::PointCloud<PointCT>::Ptr cloud)
{
	pcl::PointCloud<PointCT>::Ptr filtered(new pcl::PointCloud<PointCT>());
	copyPointCloud(*cloud, *filtered);

	for(int y = 1;y < V2_HEIGHT - 1;y++)
		for (int x = 1; x < V2_WIDTH - 1; x++)
		{

			PointCT px = cloud->at(x, y);
			vector<PointCT> npx;
			npx.push_back(cloud->at(x - 1, y - 1));
			npx.push_back(cloud->at(x - 1, y    ));
			npx.push_back(cloud->at(x - 1, y + 1));
			npx.push_back(cloud->at(x    , y - 1));
			npx.push_back(cloud->at(x    , y + 1));
			npx.push_back(cloud->at(x + 1, y - 1));
			npx.push_back(cloud->at(x + 1, y    ));
			npx.push_back(cloud->at(x + 1, y + 1));

			double max_value = 0.0f;
			for (int n = 0; n < npx.size(); n++)
			{
				double sin_apex = sin(acos(npx[n].getVector4fMap().dot(px.getVector4fMap()) / (getLength(npx[n]) * getLength(px))));
				double value = getLength(npx[n]) * sin_apex / getLength(subtract(npx[n], px));
				value = rad_to_deg(asin(value));
				if (max_value < value)
					max_value = value;
			}
			if (max_value > 170.0f)
			{
				filtered->at(x, y).x = 0.0;
				filtered->at(x, y).y = 0.0;
				filtered->at(x, y).z = 0.0;
			}
		}
	return filtered;
}

void MainEngine::CutZFilter(pcl::PointCloud<PointCT>::Ptr& cloud)
{
	for(int y = 0;y < V2_HEIGHT;y++)
		for (int x = 0; x < V2_WIDTH; x++)
		{
			if (cloud->at(x, y).z > 3.0)
			{
				cloud->at(x, y).x = 0.0;
				cloud->at(x, y).y = 0.0;
				cloud->at(x, y).z = 0.0;
			}
		}


}

void MainEngine::renderSQ() 
{
	pcd_viz->removeAllShapes();
	pcd_viz->removeAllPointClouds();
	for (int pdx = 0; pdx < JOINTS_NUM; pdx++)
	{
		pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
		PointT color;
		color.x = static_cast<double>(random_colors[pdx + 1][0]);
		color.y = static_cast<double>(random_colors[pdx + 1][1]);
		color.z = static_cast<double>(random_colors[pdx + 1][2]);

		//std::cout << sq_params[pdx].e1 << "," << sq_params[pdx].e2 << "," << sq_params[pdx].a << "," << sq_params[pdx].b << "," << sq_params[pdx].c << std::endl;
		quadEngine->GenerateSQUniformMesh(mesh, sq_params[pdx], color);

		string mesh_id = "mesh" + to_string(pdx);
		vtkSmartPointer<vtkPolyData> poly;

		pcl::io::mesh2vtk(*mesh, poly);
		pcd_viz->removePolygonMesh(mesh_id);
		pcd_viz->addModelFromPolyData(poly, mesh_id);

		pcd_viz->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, mesh_id);
	}
}

cv::Mat MainEngine::warpLabelImg(cv::Mat labelImg, cv::Mat warpImg)
{
	cv::Mat t_label = cv::Mat::zeros(warpImg.size(), CV_8U);
	for(int y = 0;y < t_label.rows;y++)
		for (int x = 0; x < t_label.cols; x++)
		{
			cv::Point2i colorCoord = warpImg.at<cv::Vec2i>(y, x);
			int colorX = static_cast<int>((1920.0 - float(colorCoord.x)) / 2.0);
			int colorY = static_cast<int>(float(colorCoord.y) / 2.0);
			
			if (colorX > 0 && colorX < 1920.0 / 2.0 && colorY > 0 && colorY < 1080.0 / 2.0)
			{
				unsigned char label = labelImg.at<unsigned char>(colorY, colorX);
				t_label.at<unsigned char>(y, x) = label;
			}
		}
	cv::flip(t_label, t_label, 1);
	return t_label;
}

void MainEngine::jointVisualization(pcl::PointCloud<PointCT>::Ptr filtered)
{

	for (int i = 0; i < joints3D.size(); i++)
		for (auto jidx = 0; jidx < joints3D[i].size(); jidx++)
		{
			PointCT px;
			px.x = joints3D[i][jidx].x;
			px.y = joints3D[i][jidx].y;
			px.z = joints3D[i][jidx].z;
			if (jidx == 0 || jidx == 1)
			{
				px.r = random_colors[1][0];
				px.g = random_colors[1][1];
				px.b = random_colors[1][2];
			}
			else if (jidx == 3) {
				px.r = random_colors[2][0];
				px.g = random_colors[2][1];
				px.b = random_colors[2][2];
			}
			else if (jidx == 4 || jidx == 5) {
				px.r = random_colors[3][0];
				px.g = random_colors[3][1];
				px.b = random_colors[3][2];
			}
			else if (jidx == 6) {
				px.r = random_colors[4][0];
				px.g = random_colors[4][1];
				px.b = random_colors[4][2];
			}
			else if (jidx == 8 || jidx == 9) {
				px.r = random_colors[5][0];
				px.g = random_colors[5][1];
				px.b = random_colors[5][2];
			}
			else if (jidx == 10) {
				px.r = random_colors[6][0];
				px.g = random_colors[6][1];
				px.b = random_colors[6][2];
			}
			else if (jidx == 12 || jidx == 13) {
				px.r = random_colors[7][0];
				px.g = random_colors[7][1];
				px.b = random_colors[7][2];
			}
			else if (jidx == 14) {
				px.r = random_colors[8][0];
				px.g = random_colors[8][1];
				px.b = random_colors[8][2];
			}
			else if (jidx == 16 || jidx == 17) {
				px.r = random_colors[9][0];
				px.g = random_colors[9][1];
				px.b = random_colors[9][2];
			}
			else if (jidx == 17 || jidx == 18) {
				px.r = random_colors[10][0];
				px.g = random_colors[10][1];
				px.b = random_colors[10][2];
			}
			filtered->push_back(px);

		}
}


void MainEngine::ActivateLoadedData()
{

	int frame = 0;
	while (loadEngine->nextDataAvailable())
	{
		std::cout << "frame " << frame << std::endl;
		//get data
		colorImage = loadEngine->getNextColorImage();
		cv::imshow("color", colorImage);
		colorDepthImage = loadEngine->getNextColorDepthImage();
		pcl::PointCloud<PointCT>::Ptr cloud(new    pcl::PointCloud<PointCT>());
		pcl::PointCloud<PointCT>::Ptr filtered(new pcl::PointCloud<PointCT>());
		cloud = loadEngine->getNextPointCloud();
		int personNum = loadEngine->getNext2D3DJoints(joints3D, joints2D);
		color2Depth = loadEngine->getNextCoordMapper();
		if (personNum == 0 || frame < 5)
		{
			frame++;
			loadEngine->readyForNextFrame();
			continue;
		}
		segmEngine->setData(colorImage);
		filtered = JumpEdgeFilter(cloud);
		CutZFilter(filtered);

		for (int i = 0; i < joints2D.size(); i++)
			for (auto jidx = 0; jidx < joints2D[i].size(); jidx++)
				if (joints2D[i][jidx].x > 0 && joints2D[i][jidx].x < colorDepthImage.cols && joints2D[i][jidx].y > 0 && joints2D[i][jidx].y < colorDepthImage.rows)
				{
					cv::circle(colorDepthImage, cv::Point(joints2D[i][jidx].x, joints2D[i][jidx].y), 4, random_colors[i], -1);
					/// joints 3D visualization
					PointCT px;
					px.x = joints3D[i][jidx].x;
					px.y = joints3D[i][jidx].y;
					px.z = joints3D[i][jidx].z;
					px.r = 255.0;
					px.g = 0.0;
					px.b = 0.0;
					//filtered->push_back(px);

				}
		//instance segmentation
		while (!segmEngine->isGet())
			continue;
		cv::Mat labelImg = segmEngine->getLabelImg();
		cv::Mat warpedLabelImg = warpLabelImg(labelImg, color2Depth);

		int personIndex = -1;
		cv::Mat instanceImg = PartsSegmentation(joints2D, warpedLabelImg, personIndex);

		///superquadric fitting
		for (int i = 0; i < partedCloud.size(); i++)
			partedCloud[i]->clear();

		cv::Mat vis_warpedImg = cv::Mat::zeros(colorDepthImage.size(), CV_8UC3);
		for (unsigned i = 0; i < warpedLabelImg.total(); ++i)
		{
			int instance_label = static_cast<int>(instanceImg.data[i]);
			if (instance_label == 0)
				continue;
			//filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).r = random_colors[instance_label][0];
			//filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).g = random_colors[instance_label][1];
			//filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).b = random_colors[instance_label][2];

			colorDepthImage.at<cv::Vec3b>(i / V2_WIDTH, i % V2_WIDTH) = random_colors[instance_label];
			vis_warpedImg.at<cv::Vec3b>(i / V2_WIDTH, i % V2_WIDTH) = random_colors[warpedLabelImg.data[i]];

			PointT px;
			px.x = filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).x;
			px.y = filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).y;
			px.z = filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).z;
			if (isnan(px.x) || isnan(px.y) || isnan(px.z))
				continue;
			if (px.x == 0.0f && px.y == 0.0f)
				continue;
			///1 origin to 0 origin
			partedCloud[instance_label - 1]->push_back(px);
		}

		if (personIndex != -1)
		{
			quadEngine->setIniTransformParams(joints3D[personIndex]);
			quadEngine->computeSQ(partedCloud);
			sq_params = quadEngine->getParams();
			renderSQ();
		}
		///visualization


		pcd_viz->addPointCloud<PointCT>(filtered, "cloud");
		//if (frame  > 50)
		//	while (!pcd_viz->wasStopped())
		//		pcd_viz->spinOnce(1);


		pcd_viz->spinOnce(1);
		pcd_viz->saveScreenshot("./result/" + to_string(frame) + ".jpg");


		cv::imshow("colorDepth", colorDepthImage);
		cv::imshow("vis_warpedImg", vis_warpedImg);
		cv::waitKey(1);
		std::cout << "visualization end" << std::endl;


		pcd_viz->removeAllPointClouds();
		frame++;
		std::cout << "render end " << std::endl;
		loadEngine->readyForNextFrame();
	}
}


void MainEngine::Activate()
{

	//viz->setBackgroundColor(1.0, 1.0, 1.0);
	int frame = 0;
	while (1)
	{
		std::cout << "frame " << frame << std::endl;
		//get data
		colorImage = kinectv2->getColorImage();
		depthImage = kinectv2->getDepthImage();
		colorDepthImage = kinectv2->getColorDepthImage();
		pcl::PointCloud<PointCT>::Ptr cloud(new pcl::PointCloud<PointCT>());
		pcl::PointCloud<PointCT>::Ptr filtered(new pcl::PointCloud<PointCT>());
		cloud = kinectv2->getRangedColorPointCloud(3.0);
		filtered = JumpEdgeFilter(cloud);
		
		//instance segmentation
		//segmEngine->execute(colorImage);
		cv::Mat labelImg = segmEngine->getLabelImg();
		int personNum = kinectv2->getBodyPoints(joints2D, joints3D);

		for (int i = 0; i < joints2D.size(); i++)
		{
			for (auto jidx = 0; jidx <joints2D[i].size(); jidx++)
				if(joints2D[i][jidx].x >= 0 && joints2D[i][jidx].x < colorDepthImage.cols && joints2D[i][jidx].y >= 0 && joints2D[i][jidx].y < colorDepthImage.rows)
					circle(colorDepthImage, cv::Point(joints2D[i][jidx].x, joints2D[i][jidx].y), 4, random_colors[i], -1);
		}


		//
		//cv::Mat warpedLabelImg = kinectv2->MapLabelImageToDepth(labelImg);

		//int personIndex = -1;
		//cv::Mat instanceImg = PartsSegmentation(joints2D, warpedLabelImg, personIndex);

		////superquadric fitting
		//for (int i = 0; i < partedCloud.size(); i++)
		//	partedCloud[i]->clear();

		//std::cout << "PointCloud to label " << std::endl;
		//for (unsigned i = 0; i < warpedLabelImg.total(); ++i)
		//{
		//	int instance_label = static_cast<int>(instanceImg.data[i]);
		//	if (instance_label == 0)
		//		continue;
		//	filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).r = random_colors[instance_label][0];
		//	filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).g = random_colors[instance_label][1];
		//	filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).b = random_colors[instance_label][2];

		//	PointT px;
		//	px.x = filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).x;
		//	px.y = filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).y;
		//	px.z = filtered->at(V2_WIDTH - i % V2_WIDTH - 1, i / V2_WIDTH).z;
		//	if (isnan(px.x) || isnan(px.y) || isnan(px.z))
		//		continue;
		//	if (px.x == 0.0f && px.y == 0.0f)
		//		continue;
		//	//1 origin to 0 origin
		//	partedCloud[instance_label - 1]->push_back(px);
		//}


		//if (personIndex != -1 && false)
		//{
		//	quadEngine->setIniTransformParams(joints3D[personIndex]);
		//	quadEngine->computeSQ(partedCloud);
		//	sq_params = quadEngine->getParams();
		//	renderSQ();
		//}

		//if (personIndex != -1)
		//{
		//	for (int pdx = 0; pdx < joints3D[personIndex].size(); pdx++)
		//	{
		//		PointCT px;
		//		px.x = joints3D[personIndex][pdx].x;
		//		px.y = joints3D[personIndex][pdx].y;
		//		px.z = joints3D[personIndex][pdx].z;
		//		px.r = 255.0;
		//		px.g = 0.0;
		//		px.b = 0.0;
		//		filtered->push_back(px);
		//	}

		//	io::savePCDFileBinaryCompressed("./tmp/" + to_string(frame) + ".pcd", *filtered);
		//}
		//

		////visualization
		//viz->addPointCloud<PointCT>(filtered, "cloud");
		cv::imshow("colorDepth", colorDepthImage);
		cv::waitKey(1);
		//viz->spinOnce(1);

		//viz->removeAllPointClouds();
		frame++;
		std::cout << "render end " << std::endl;
	}
}
