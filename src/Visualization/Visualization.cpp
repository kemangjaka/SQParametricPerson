#include "Visualization.h"


Visualization::Visualization()
{
	pcl::visualization::PCLVisualizer pcd_viz("test visualizer");
	//pcd_viz.setBackgroundColor(1.0, 1.0, 1.0);
	while (!pcd_viz.wasStopped())
		pcd_viz.spinOnce(1);
}


Visualization::~Visualization()
{

}