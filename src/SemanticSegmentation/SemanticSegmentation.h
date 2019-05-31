#pragma once

#include <opencv2/opencv.hpp>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <thread>
class SemSeg {
public:
	SemSeg();
	~SemSeg();
	// Warning, getPyObject requiers a decref:
	cv::Mat getLabelImg();
	bool isGet();
	void setData(cv::Mat img);
	bool isReady();

private:
	void startThreadLoop();
	void loop();
	PyObject* createArguments(cv::Mat rgbImage);
	void* loadModule();
	void initialise();
	inline PyObject* getPyObject(const char* name);
	cv::Mat extractImage();
	void execute();

	PyObject *pModule;
	PyObject *pExecute;
	cv::Mat maskImg; 
	cv::Mat inputImg;
	std::thread thread;
	bool _isSet = false;
	bool _isGet = false;
	bool _isReady = false;
};