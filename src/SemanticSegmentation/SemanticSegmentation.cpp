#include "SemanticSegmentation.h"
#include <windows.h>

void usleep(__int64 usec)
{
	HANDLE timer;
	LARGE_INTEGER ft;

	ft.QuadPart = -(10 * usec); // Convert to 100 nanosecond interval, negative value indicates relative time

	timer = CreateWaitableTimer(NULL, TRUE, NULL);
	SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
	WaitForSingleObject(timer, INFINITE);
	CloseHandle(timer);
}
SemSeg::SemSeg()
{
	startThreadLoop();
}

SemSeg::~SemSeg()
{

}

void SemSeg::startThreadLoop()
{
	if (thread.get_id() == std::thread::id())
	{
		std::cout << "start thread ..." << std::endl;
		thread = std::thread(&SemSeg::loop, this);
	}
}

void SemSeg::loop() {
	initialise();
	_isReady = true;
	while (1)
	{
		while (!_isSet)
		{
			
			usleep(1e3);
			continue;
		}
		_isSet = false;
		execute();
		_isGet = true;
	}

}

void SemSeg::initialise()
{

	Py_SetProgramName((wchar_t*)L"Execute_RefineNet");
	Py_Initialize();
	wchar_t const * argv2[] = { L"Execute_RefineNet.py" };
	PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

	// Load module
	loadModule();

	// Get function
	pExecute = PyObject_GetAttrString(pModule, "execute");
	if (pExecute == NULL || !PyCallable_Check(pExecute)) {
		if (PyErr_Occurred()) {
			std::cout << "Python error indicator is set:" << std::endl;
			PyErr_Print();
		}
		throw std::runtime_error("Could not load function 'execute' from MaskRCNN module.");
	}
	std::cout << "* Initialised Done" << std::endl;
}

void* SemSeg::loadModule()
{
	std::cout << " * Loading module..." << std::endl;
	pModule = PyImport_ImportModule("Execute_RefineNet");
	if (pModule == NULL) {
		if (PyErr_Occurred()) {
			std::cout << "Python error indicator is set:" << std::endl;
			PyErr_Print();
		}
		throw std::runtime_error("Could not open MaskRCNN module.");
	}
	import_array();
	return 0;
}

cv::Mat SemSeg::extractImage() {
	PyObject* pImage = getPyObject("result");
	PyArrayObject *pImageArray = (PyArrayObject*)(pImage);
	//assert(pImageArray->flags & NPY_ARRAY_C_CONTIGUOUS);

	unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pImageArray, 0);
	npy_intp h = PyArray_DIM(pImageArray, 0);
	npy_intp w = PyArray_DIM(pImageArray, 1);

	cv::Mat result;
	cv::Mat(h, w, CV_8UC1, pData).copyTo(result);
	Py_DECREF(pImage);
	return result;
}

PyObject *SemSeg::getPyObject(const char* name){
    PyObject* obj = PyObject_GetAttrString(pModule, name);
    if(!obj || obj == Py_None) throw std::runtime_error(std::string("Failed to get python object: ") + name);
    return obj;
}

void SemSeg::execute()
{
	//cv::cvtColor(img, img, CV_RGBA2RGB);	
	Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, createArguments(inputImg), NULL));
	maskImg = extractImage();

}

void SemSeg::setData(cv::Mat img)
{
	inputImg = img.clone();
	_isSet = true;
}

cv::Mat SemSeg::getLabelImg()
{
	_isGet = false;
	return maskImg;
}

bool SemSeg::isGet()
{
	return _isGet;
}

bool SemSeg::isReady()
{
	return _isReady;
}

PyObject *SemSeg::createArguments(cv::Mat rgbImage) {
	assert(rgbImage.channels() == 3);
	npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
	return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); // TODO Release?
}