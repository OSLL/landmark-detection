cmake_minimum_required(VERSION 2.8.9)

project(landmark_detector)

option(WITH_ZBAR "use zbar library as qrcode reader" ON)

if(NOT DEFINED OpenCV_LIBS)
	find_package(OpenCV REQUIRED)
endif()

add_subdirectory(detector)
add_subdirectory(detector_gui)