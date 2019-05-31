# Example CMakeLists.txt
# FindKInectSDK.cmake copy to CMake\share\cmake-3.5\Modules or same directory as this file

cmake_minimum_required( VERSION 2.8 )
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(${CMAKE_CURRENT_SOURCE_DIR})
project( solution )
add_executable( project src/Main/Source.cpp )
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )
find_package( KinectSDK2 REQUIRED )

  include_directories( ${KinectSDK2_INCLUDE_DIRS} )
  link_directories( ${KinectSDK2_LIBRARY_DIRS} )
  target_link_libraries( project ${KinectSDK2_LIBRARIES} )
  add_custom_command( TARGET project POST_BUILD ${KinectSDK2_COMMANDS} )


find_package( PCL 1.8 REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )
target_link_libraries( project ${PCL_LIBRARIES} )

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories( ${OpenCV_LIBRARY_DIRS} )
target_link_libraries( project ${OpenCV_LIBRARIES} )


find_package (Ceres REQUIRED)
include_directories (${CERES_INCLUDES})
link_directories( ${CERES_LIBRARY_DIRS} )
target_link_libraries( project ${CERES_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Using python:  ${PYTHONLIBS_VERSION_STRING} ")
message("Using PYTHON_LIBRARIES:  ${PYTHON_LIBRARIES} ")
message("Using PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS} ")
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(${PYTHON_LIBRARIES})

include_directories("C:/Program Files/Anaconda3/Lib/site-packages/numpy/core/include")
#target_link_libraries(${NUMPY_LIBRARY})