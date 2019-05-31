#pragma once

#include "../Main/Header.h"

inline double getLength(PointCT px)
{
	return sqrt(pow(px.x, 2.0) + pow(px.y, 2.0) + pow(px.z, 2.0));
}


inline PointCT subtract(PointCT p1, PointCT p2)
{
	PointCT ans;
	ans.x = p1.x - p2.x;
	ans.y = p1.y - p2.y;
	ans.z = p1.z - p2.z;
	return ans;
}
inline PointT subtract(PointT p1, PointT p2)
{
	PointT ans;
	ans.x = p1.x - p2.x;
	ans.y = p1.y - p2.y;
	ans.z = p1.z - p2.z;
	return ans;
}

inline PointT mean(PointT p1, PointT p2)
{
	PointT ans;
	ans.x = (p1.x + p2.x) / 2.0f;
	ans.y = (p1.y + p2.y) / 2.0f;
	ans.z = (p1.z + p2.z) / 2.0f;
	return ans;
}

inline PointT mean(PointT p1, PointT p2, PointT p3)
{
	PointT ans;
	ans.x = (p1.x + p2.x + p3.x) / 3.0f;
	ans.y = (p1.y + p2.y + p3.y) / 3.0f;
	ans.z = (p1.z + p2.z + p3.z) / 3.0f;
	return ans;
}

#define rad_to_deg(rad) (((rad)/2/M_PI)*360)
