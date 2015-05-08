#include "mainwindow.h"
#include <QDebug>
#include <QPainter>
#include <QRadioButton>
#include <QButtonGroup>
#include <QFileDialog>
#include <QGridLayout>
#include <QApplication>
#include <QFrame>
#include <QThread>
#include <qmath.h>

#include "opencv2/imgproc/imgproc.hpp"

#ifdef PERF_TEST
#include <iostream>
#endif

MainWindow::MainWindow(const QString &sdir, const QString &detectorParamsPath, float focalLength, float objectSize, QWidget *parent) : QMainWindow(parent), fpsCounter(0), staticImageEnabled(false), shadersDir(sdir) {
#ifdef WITH_GPU
    detector.createGPUWorker(640, 480, this);
    if(shadersDir != "/" && shadersDir.lastIndexOf('/') != shadersDir.size() - 1) shadersDir += "/";
#else
    outRed = new ImageLabel(this);
    outRed->setMinimumSize(10, 10);
#endif
    outCap = new ImageLabel(this);
    outGreen = new ImageLabel(this);
    outBlue = new ImageLabel(this);
    outCap->setMinimumSize(10, 10);
    outGreen->setMinimumSize(10, 10);
    outBlue->setMinimumSize(10, 10);

    lbThreshold = new QLabel(this);
    lbRatio = new QLabel(this);
    lbDist = new QLabel(this);
    lbSat = new QLabel(this);

    cbTracking = new QCheckBox("Tracking", this);
    cbTracking->setChecked(true);

    cbShowDetected = new QCheckBox("Show detection status", this);

    slThreshold = new QSlider(Qt::Horizontal, this);
    slThreshold->setRange(0, 255);
    connect(slThreshold, SIGNAL(valueChanged(int)), this, SLOT(setThreshold(int)));
    slThreshold->setValue(100);

    slRatio = new QSlider(Qt::Horizontal, this);
    slRatio->setRange(0, 200);
    connect(slRatio, SIGNAL(valueChanged(int)), this, SLOT(setRatio(int)));
    slRatio->setValue(60);

    slDist = new QSlider(Qt::Horizontal, this);
    slDist->setRange(0, 640);
    connect(slDist, SIGNAL(valueChanged(int)), this, SLOT(setDistThreshold(int)));
    slDist->setValue(240);

    slSatThreshold = new QSlider(Qt::Horizontal, this);
    slSatThreshold->setRange(0, 255);
    connect(slSatThreshold, SIGNAL(valueChanged(int)), this, SLOT(setSatThreshold(int)));
    slSatThreshold->setValue(100);

    pbRunDetector = new QPushButton("Run", this);
    connect(pbRunDetector, SIGNAL(clicked()), this, SLOT(runNTimes()));

    slTries = new QSlider(Qt::Horizontal, this);
    slTries->setRange(1, 50);
    connect(slTries, SIGNAL(valueChanged(int)), this, SLOT(setNumDetections(int)));
    slTries->setValue(5);

    slCA = new QSlider(Qt::Horizontal, this);
    slCA->setRange(1, 31);
    connect(slCA, SIGNAL(valueChanged(int)), this, SLOT(setCornerParams()));

    slCB = new QSlider(Qt::Horizontal, this);
    slCB->setRange(1, 50);
    connect(slCB, SIGNAL(valueChanged(int)), this, SLOT(setCornerParams()));

    slCK = new QSlider(Qt::Horizontal, this);
    slCK->setRange(1, 100);
    connect(slCK, SIGNAL(valueChanged(int)), this, SLOT(setCornerParams()));

    slCT = new QSlider(Qt::Horizontal, this);
    slCT->setRange(0, 255);
    connect(slCT, SIGNAL(valueChanged(int)), this, SLOT(setCornerParams()));

    lbCA = new QLabel(this);
    lbCB = new QLabel(this);
    lbCK = new QLabel(this);
    lbCT = new QLabel(this);

    slCA->setValue(3);
    slCB->setValue(9);
    slCK->setValue(4);
    slCT->setValue(175);

    QGridLayout *sl = new QGridLayout();
    sl->addWidget(new QLabel("Canny", this), 0, 0);
    sl->addWidget(slThreshold, 0, 1);
    sl->addWidget(lbThreshold, 0, 2);
    sl->addWidget(new QLabel("Ratio:", this), 1, 0);
    sl->addWidget(slRatio, 1, 1);
    sl->addWidget(lbRatio, 1, 2);
    sl->addWidget(new QLabel("Dist:", this), 2, 0);
    sl->addWidget(slDist, 2, 1);
    sl->addWidget(lbDist, 2, 2);
    sl->addWidget(new QLabel("Sat:", this), 3, 0);
    sl->addWidget(slSatThreshold, 3, 1);
    sl->addWidget(lbSat, 3, 2);

    sl->addWidget(new QLabel("Corner B:", this), 4, 0);
    sl->addWidget(slCB, 4, 1);
    sl->addWidget(lbCB, 4, 2);
    sl->addWidget(new QLabel("Corner A:", this), 5, 0);
    sl->addWidget(slCA, 5, 1);
    sl->addWidget(lbCA, 5, 2);
    sl->addWidget(new QLabel("Corner K:", this), 6, 0);
    sl->addWidget(slCK, 6, 1);
    sl->addWidget(lbCK, 6, 2);
    sl->addWidget(new QLabel("Corner T:", this), 7, 0);
    sl->addWidget(slCT, 7, 1);
    sl->addWidget(lbCT, 7, 2);

    sl->setContentsMargins(0, 0, 0, 0);
    sl->setSpacing(5);
    sl->setColumnMinimumWidth(1, 100);

    QRadioButton *rbCap = new QRadioButton("Camera", this);
    QRadioButton *rbImg = new QRadioButton("Image", this);
    rbCap->setObjectName("rb_capture");

    QButtonGroup *bgImgSrc = new QButtonGroup(this);
    bgImgSrc->addButton(rbCap, 0);
    bgImgSrc->addButton(rbImg, 1);
    connect(bgImgSrc, SIGNAL(buttonToggled(int,bool)), this, SLOT(changeImageSource(int,bool)));

    pbSelImg = new QPushButton("...", this);
    connect(pbSelImg, SIGNAL(clicked()), this, SLOT(loadImage()));

    QHBoxLayout *imgl = new QHBoxLayout();
    imgl->addWidget(rbImg);
    imgl->addWidget(pbSelImg);
    imgl->setSpacing(5);
    imgl->setContentsMargins(0, 0, 0, 0);

    lbDetected = new QLabel(this);
    lbDetected->setMaximumWidth(22);
    lbX = new QLabel("X: ", this);
    lbY = new QLabel("Y: ", this);
    lbZ = new QLabel("Z: ", this);
    lbImg = new QLabel(this);
    QHBoxLayout *allx = new QHBoxLayout();
    allx->addWidget(lbX, 1);
    allx->addWidget(lbDetected);

    QHBoxLayout *dtl = new QHBoxLayout();
    dtl->addWidget(new QLabel("N:"));
    dtl->addWidget(slTries, 1);
    dtl->addWidget(pbRunDetector);

    QFrame *hline = new QFrame(this);
    hline->setFrameStyle(QFrame::HLine | QFrame::Sunken);

    QVBoxLayout *all = new QVBoxLayout();
    all->addWidget(cbTracking);
    all->addWidget(cbShowDetected);
    all->addLayout(allx);
    all->addWidget(lbY);
    all->addWidget(lbZ);
    all->addWidget(lbImg);
    all->addWidget(hline);
    all->addLayout(dtl);
    all->setContentsMargins(0, 0, 0, 0);
    all->setSpacing(5);

    QVBoxLayout *cl = new QVBoxLayout();
    cl->addLayout(sl);
    cl->addWidget(new QLabel("Image source:"));
    cl->addWidget(rbCap);
    cl->addLayout(imgl);
    cl->addLayout(all);
    cl->addStretch(1);
    cl->setSpacing(5);
    cl->setContentsMargins(0, 0, 0, 0);

    QWidget *w = new QWidget(this);
    QGridLayout *l = new QGridLayout();
    l->addWidget(outCap, 0, 0);
#ifdef WITH_GPU
    l->addWidget(detector.gpuWorker, 0, 1);
#else
    l->addWidget(outRed, 0, 1);
#endif
    l->addWidget(outGreen, 1, 0);
    l->addWidget(outBlue, 1, 1);
    l->addLayout(cl, 0, 2, 2, 1);
    l->setContentsMargins(5, 5, 5, 5);
    l->setSpacing(5);
    l->setColumnStretch(0, 1);
    l->setColumnStretch(1, 1);
    l->setColumnStretch(2, 0);
    w->setLayout(l);

    this->setCentralWidget(w);

    cv::FileStorage fs(detectorParamsPath.toStdString(), cv::FileStorage::READ);
    if(fs.isOpened()) {
        cv::Mat cm, dc;
        fs["CameraMatrix"] >> cm;
        fs["DistortionCoeffs"] >> dc;
        if(!cm.empty() && !dc.empty()) detector.setCameraParams(cm, dc, focalLength, objectSize);
        else qDebug() << "unable to load camera matrix";
    } else qDebug() << "unable to load file storage";

    frameTimer = new QTimer(this);
    connect(frameTimer, SIGNAL(timeout()), this, SLOT(readFromCameraAndProcess()));

    fpsTimer = new QTimer(this);
    connect(fpsTimer, SIGNAL(timeout()), this, SLOT(showFPS()));
    fpsTimer->start(1000);

    QTimer::singleShot(10, this, SLOT(startDetector()));

    resize(825, 495);
}

MainWindow::~MainWindow() {
}

void MainWindow::startDetector() {
#ifdef WITH_GPU
    assert(detector.initGPU(shadersDir.toStdString() + "blur.fsh",
                            shadersDir.toStdString() + "hsv.fsh",
                            shadersDir.toStdString() + "sobel.fsh",
                            shadersDir.toStdString() + "canny.fsh"));
#endif
    findChild<QRadioButton*>("rb_capture")->setChecked(true);
}

#define CENTER(r) QPoint(r.x + r.width / 2, r.y + r.height / 2)
#define TO_QRECT(r) QRect((r).x, (r).y, (r).width, (r).height)

void MainWindow::showFPS() {
    lbZ->setText(QString("FPS: %1").arg(fpsCounter));
    fpsCounter = 0;
}

void MainWindow::setThreshold(int val) {
    detector.setCannyThreshold(val);
    lbThreshold->setText(QString::number(val));
    if(staticImageEnabled) processFrame(staticImage);
}

void MainWindow::setRatio(int val) {
    float colorRatio = 1 + (float)val / 100.0;
    detector.setColorRatio(colorRatio);
    lbRatio->setText(QString::number(colorRatio));
    if(staticImageEnabled) processFrame(staticImage);
}

void MainWindow::setDistThreshold(int val) {
    detector.setDistThreshold(val * val);
    lbDist->setText(QString::number(val));
    if(staticImageEnabled) processFrame(staticImage);
}

void MainWindow::setSatThreshold(int val) {
    detector.setSaturationThreshold(val);
    lbSat->setText(QString::number(val));
    if(staticImageEnabled) processFrame(staticImage);
}

void MainWindow::setNumDetections(int val) {
    pbRunDetector->setText(QString("Run %1").arg(val, 2));
}

void MainWindow::setCornerParams() {
    lbCA->setText(QString::number(slCA->value()));
    lbCB->setText(QString::number(slCB->value()));
    lbCK->setText(QString::number(slCK->value() / 100.0));
    lbCT->setText(QString::number(slCT->value()));
    detector.setCornerDetectorParams(slCA->value(), slCB->value(), slCK->value(), slCT->value());
}

void MainWindow::changeImageSource(int src, bool enabled) {
    if(!enabled) return;
    switch(src) {
    case 0: //camera
        staticImageEnabled = false;
        pbSelImg->setEnabled(false);
        cap = cv::VideoCapture(0);
//        cap.set(CV_CAP_PROP_CONVERT_RGB, false);
        frameTimer->start(100);
        break;
    case 1: //image
        staticImageEnabled = true;
        pbSelImg->setEnabled(true);
        frameTimer->stop();
        cap.release();
        if(!staticImage.empty()) processFrame(staticImage);
        break;
    default:
        break;
    }
}

void MainWindow::loadImage() {
    QString fileName = QFileDialog::getOpenFileName(this);
    if(fileName.isEmpty()) return;
    staticImage = cv::imread(fileName.toStdString());
    if(staticImage.empty()) {
        qDebug() << "Unable to read image";
    } else {
        processFrame(staticImage);
    }
    /*{
        while(true) {
            qApp->processEvents();
            processFrame(staticImage);
            ++fpsCounter;
        }
    }*/
}

void MainWindow::readFromCameraAndProcess() {
    cv::Mat frame;
    if(!cap.read(frame)) {
        qDebug() << "unable to read frame";
        return;
    }
    processFrame(frame);
/*    frameTimer->stop();
    while(true) {
        qApp->processEvents();
        cv::Mat frame;
        if(!cap.read(frame)) {
            qDebug() << "unable to read frame";
            return;
        }
//        ++fpsCounter;
        processFrame(frame);
    }
*/
}

void MainWindow::runNTimes() {
    if(staticImageEnabled) return;
    frameTimer->stop();
    int detectedCount = 0;
    for(int i = 0; i < slTries->value(); ++i) {
        cv::Mat frame;
        if(!cap.read(frame)) {
            qDebug() << "unable to read frame";
            return;
        }
        std::vector<LabelDescriptor> lds = detector.detect(frame);
        if(!lds.empty()) ++detectedCount;
        qApp->thread()->msleep(100);
    }
    lbY->setText(QString("Detection rate: %1%").arg(detectedCount * 100 / slTries->value()));
    frameTimer->start();
}

//=================================================================================================

#define RECT_CENTER(r) cv::Point2i((r).x + (r).width / 2, (r).y + (r).height / 2)

static QPixmap rectPixmap(Qt::GlobalColor color, int w = 16, int h = 16) {
    QPixmap dt(w, h);
    QPainter dtp(&dt);
    dtp.fillRect(dt.rect(), color);
    dtp.end();
    return dt;
}

void MainWindow::processFrame(const cv::Mat &srcFrame) {
    cv::Mat cframe;
    srcFrame.copyTo(cframe);

    std::vector<cv::Rect> redRects, blueRects;
    std::vector<LabelDescriptor> lds = detector.detect(cframe, &blueRects, &redRects, true);
    cv::cvtColor(cframe, cframe, CV_BGR2RGB);

    if(cbShowDetected->isChecked()) {
        lbDetected->setPixmap(rectPixmap(lds.empty() ? Qt::red : Qt::green));
    } else if(!lds.empty()) {
        cv::Mat warped;
        if(detector.readLabel(srcFrame, detector.edgesOut, lds[0], &warped)) {
            lbX->setText(QString::fromStdString(lds[0].data));
            lbDetected->setPixmap(rectPixmap(Qt::green));
        } else {
            lbDetected->setPixmap(rectPixmap(Qt::red));
        }        
        outBlue->setPixmap(QPixmap::fromImage(QImage(warped.data, warped.cols, warped.rows, QImage::Format_Indexed8)));
        for(std::vector<cv::Point2f>::iterator p = lds[0].corners.begin(); p != lds[0].corners.end(); ++p) {
            cv::circle(cframe, *p, 10, cv::Scalar(0, 255, 0), 2);
        }
        if(detector.estimatePosition(lds[0])) {
            lbY->setText(QString("x: %1 | y: %2 | z: %3 | d: %4")
                                .arg((int)(lds[0].xAngle / M_PI * 180))
                                .arg((int)(lds[0].yAngle / M_PI * 180))
                                .arg((int)(lds[0].zAngle / M_PI * 180))
                                .arg((int)lds[0].distance)
                        );
        }
    }

    if(cbTracking->isChecked()) {
        detector.trackObjects(lds);
        const std::vector<LabelDescriptor> &objects = detector.getObjects();

        for(std::vector<LabelDescriptor>::const_iterator obj = objects.begin(); obj != objects.end(); ++obj) {
            if(obj->lifetime == 0) {
                cv::rectangle(cframe, obj->br, cv::Scalar(0, 255, 0), 2);
                cv::putText(cframe, QString("id %1").arg(obj->id).toStdString(), obj->br.tl() + cv::Point2i(5, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0), 2);
            }
        }
    } else {
        for(std::vector<cv::Rect>::const_iterator r = blueRects.begin(); r != blueRects.end(); ++r) {
            cv::rectangle(cframe, *r, cv::Scalar(0, 0, 255), 2);
        }
        for(std::vector<cv::Rect>::const_iterator r = redRects.begin(); r != redRects.end(); ++r) {
            cv::rectangle(cframe, *r, cv::Scalar(255, 0, 0), 2);
        }

        for(std::vector<LabelDescriptor>::iterator d = lds.begin(); d != lds.end(); ++d) {
            for(int i = 0; i < 3; ++i) {
                cv::rectangle(cframe, d->fips[i], cv::Scalar(0, 0, 255), 2);
                cv::line(cframe, RECT_CENTER(d->fips[i]), RECT_CENTER(d->fips[(i + 1) % 3]), cv::Scalar(0, 255, 0), 2);
            }
            cv::rectangle(cframe, d->ap, cv::Scalar(255, 0, 0), 2);
            cv::line(cframe, RECT_CENTER(d->fips[0]), RECT_CENTER(d->ap), cv::Scalar(0, 255, 0), 2);
        }
    }

    outCap->setPixmap(QPixmap::fromImage(QImage(cframe.data, cframe.cols, cframe.rows, QImage::Format_RGB888)));
    outGreen->setPixmap(QPixmap::fromImage(QImage(detector.edgesOut.data, detector.edgesOut.cols, detector.edgesOut.rows, QImage::Format_Indexed8)));
//    outBlue->setPixmap(QPixmap::fromImage(QImage(detector.gpuWorker->renderOutput.data, detector.gpuWorker->renderOutput.cols, detector.gpuWorker->renderOutput.rows, QImage::Format_RGB888)));
#ifndef WITH_GPU
    outRed->setPixmap(QPixmap::fromImage(QImage(detector.satOut.data, detector.satOut.cols, detector.satOut.rows, QImage::Format_Indexed8)));
#endif
#ifdef PERF_TEST
    std::cout << "---------------------------------------" << std::endl;
#endif
//    outBlue->setPixmap(QPixmap::fromImage(QImage(rgbChannels[2].data, rgbChannels[2].cols, rgbChannels[2].rows, QImage::Format_Indexed8)));
//    outRed->setPixmap(QPixmap::fromImage(QImage(rgbChannels[0].data, rgbChannels[0].cols, rgbChannels[0].rows, QImage::Format_Indexed8)));
//    outGreen->setPixmap(QPixmap::fromImage(QImage(rgbChannels[1].data, rgbChannels[1].cols, rgbChannels[1].rows, QImage::Format_Indexed8)));

    ++fpsCounter;
}

//=================================================================================================

ImageLabel::ImageLabel(QWidget *parent) : QWidget(parent) {
}

void ImageLabel::setPixmap(const QPixmap &p) {
    pix = p;
    update();
}

QPixmap *ImageLabel::pixmap() {
    return &pix;
}

void ImageLabel::paintEvent(QPaintEvent *e) {
    QWidget::paintEvent(e);

    if(pix.isNull()) return;

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.drawPixmap(0, 0, pix.scaled(e->rect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
