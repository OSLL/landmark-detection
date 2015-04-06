#include "mainwindow.h"
#include <QApplication>

//usage: ./alt_detector_gui [camera params] [focal length in pix] [object size in mm]
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    QString detectorParamsPath = "detector_params.xml";
    float objectSize = 90.0, focalLength = 693.1;
    if(argc > 1) detectorParamsPath = argv[1];
    if(argc > 2) focalLength = QString(argv[2]).toFloat();
    if(argc > 3) objectSize = QString(argv[3]).toFloat();
    MainWindow w(detectorParamsPath, focalLength, objectSize);
    w.show();

    return a.exec();
}
