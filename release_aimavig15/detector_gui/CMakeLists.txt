cmake_minimum_required(VERSION 2.8.9)

project(detector_gui)

if(WITH_ZBAR)
	find_library(ZBAR_LIB NAMES zbar libzbar)
	find_path(ZBAR_INCLUDE_DIR zbar.h)
	
	if(NOT ZBAR_LIB OR NOT ZBAR_INCLUDE_DIR)
		message(SEND_ERROR "ZBar library not found")
		return() 
	endif()

	include_directories(${ZBAR_INCLUDE_DIR})	
	add_definitions(-DWITH_ZBAR)
else()
	set(ZBAR_LIB "")
endif()

if(WITH_PERF)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	add_definitions(-DPERF_TEST)
    add_definitions(-DUSE_QT_PERF_TOOLS)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(DETECTOR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../detector)

include_directories(${OpenCV_INCLUDE_DIRS} ${DETECTOR_SRC_DIR})

add_definitions(-DDETECTOR_DEBUG)
if(WITH_GPU)
    add_definitions(-DWITH_GPU)
    add_library(detector ${DETECTOR_SRC_DIR}/br_detector.cpp ${DETECTOR_SRC_DIR}/br_detector.h ${DETECTOR_SRC_DIR}/hungarian.c ${DETECTOR_SRC_DIR}/hungarian.h gpu_worker_win.cpp gpu_worker_win.h)
else()
    add_library(detector ${DETECTOR_SRC_DIR}/br_detector.cpp ${DETECTOR_SRC_DIR}/br_detector.h ${DETECTOR_SRC_DIR}/hungarian.c ${DETECTOR_SRC_DIR}/hungarian.h)
endif()

add_executable(alt_detector_gui WIN32 main.cpp mainwindow.cpp)

target_link_libraries(alt_detector_gui detector ${OpenCV_LIBS} ${ZBAR_LIB})
if(WIN32)
    target_link_libraries(alt_detector_gui iconv)
endif()

qt5_use_modules(alt_detector_gui Widgets)
if(WITH_PERF OR WITH_GPU)
    qt5_use_modules(detector Core Widgets)
endif()
