cmake_minimum_required(VERSION 2.8.9)

project(detector_gui)

option(WITH_ZBAR "use zbar library as qrcode reader" ON)

find_package(Qt5Widgets REQUIRED)

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

include_directories(../detector)
add_definitions(-DDETECTOR_DEBUG)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(WIN32)
	add_executable(detector_gui WIN32 main.cpp mainwindow.cpp)
else()
	add_executable(detector_gui main.cpp mainwindow.cpp)
endif()

target_link_libraries(detector_gui detector ${OpenCV_LIBS} ${ZBAR_LIB})
add_dependencies(detector_gui detector)
if(WIN32)
	target_link_libraries(detector_gui iconv)
endif()

qt5_use_modules(detector_gui Widgets)

