#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <QSlider>
#include <QCheckBox>
#include <QPushButton>
#include <QPaintEvent>

#include "opencv2/highgui/highgui.hpp"

#include "br_detector.h"

class ImageLabel : public QWidget {
    Q_OBJECT

public:
    ImageLabel(QWidget *parent);

    void setPixmap(const QPixmap &p);
    QPixmap *pixmap();

protected:
    void paintEvent(QPaintEvent *e);

private:
    QPixmap pix;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(const QString &detectorParamsPath, float focalLength, float objectSize, QWidget *parent = 0);
    ~MainWindow();

private slots:
    void processFrame(const cv::Mat &srcFrame);
    void setThreshold(int val);
    void setRatio(int val);
    void setDistThreshold(int val);
    void setSatThreshold(int val);
    void setNumDetections(int val);
    void changeImageSource(int src, bool enabled);
    void readFromCameraAndProcess();
    void loadImage();
    void showFPS();
    void runNTimes();
    void setCornerParams();

private:
    cv::VideoCapture cap;
    QTimer *frameTimer, *fpsTimer;
    ImageLabel *outCap, *outRed, *outGreen, *outBlue;
    QLabel *lbThreshold, *lbRatio, *lbDist, *lbSat;
    QLabel *lbX, *lbY, *lbZ, *lbImg, *lbDetected;
    QSlider *slThreshold, *slRatio, *slDist, *slSatThreshold, *slTries;
    QPushButton *pbSelImg, *pbRunDetector;
    QCheckBox *cbTracking, *cbShowDetected;
    QSlider *slCA, *slCB, *slCK, *slCT;
    QLabel *lbCA, *lbCB, *lbCK, *lbCT;

    int fpsCounter;
    cv::Mat staticImage;
    bool staticImageEnabled;

    BRDetector detector;

};

#endif // MAINWINDOW_H
