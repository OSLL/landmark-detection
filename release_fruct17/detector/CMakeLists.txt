cmake_minimum_required(VERSION 2.8.9)

project(br_detector)

if(WITH_ZBAR)
	find_library(ZBAR_LIB NAMES zbar libzbar)
	find_path(ZBAR_INCLUDE_DIR zbar.h)
	
	if(NOT ZBAR_LIB OR NOT ZBAR_INCLUDE_DIR)
		message(SEND_ERROR "ZBar library not found")
		return() 
	endif()

	include_directories(${ZBAR_INCLUDE_DIR})	
	add_definitions(-DWITH_ZBAR)
endif()

include_directories(${OpenCV_INCLUDE_DIRS} .)

add_definitions(-DDETECTOR_DEBUG)
add_library(detector br_detector.cpp br_detector.h hungarian.c hungarian.h)
