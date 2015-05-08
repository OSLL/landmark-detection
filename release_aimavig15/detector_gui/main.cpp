#include "mainwindow.h"
#include <QApplication>

#include <iostream>

//usage: ./alt_detector_gui [shaders directory] [camera params] [focal length in pix] [object size in mm]
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    QString detectorParamsPath = "calib_example.xml", shadersDir = ".";
    float objectSize = 90.0, focalLength = 693.1;

#ifndef WITH_GPU
    if(argc > 1) detectorParamsPath = argv[1];
    if(argc > 2) focalLength = QString(argv[2]).toFloat();
    if(argc > 3) objectSize = QString(argv[3]).toFloat();
#else
    if(argc < 2) {
        std::cout << "usage: " << argv[0] << " <shaders directory> [camera params] [focal length in pix] [object size in mm]" << std::endl;
        return -1;
    }
    shadersDir = argv[1];
    if(argc > 2) detectorParamsPath = argv[2];
    if(argc > 3) focalLength = QString(argv[3]).toFloat();
    if(argc > 4) objectSize = QString(argv[4]).toFloat();
#endif
    MainWindow w(shadersDir, detectorParamsPath, focalLength, objectSize);
    w.show();

    return a.exec();
}
