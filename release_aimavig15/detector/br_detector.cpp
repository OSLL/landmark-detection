#include "br_detector.h"
#include "hungarian.h"

#include <map>
#include <algorithm>
#include <cmath>

#include "opencv2/calib3d/calib3d.hpp"

#ifdef WITH_ZBAR
#include "zbar.h"
#endif

#ifdef PERF_TEST
#ifndef USE_QT_PERF_TOOLS
#include <chrono>
#include <iostream>
#define TIMER_NAME(timer) #timer
#define TIMER_INIT(timer) \
    std::chrono::high_resolution_clock::time_point timer##_start_time; \
    std::chrono::duration<double> timer##_duration;
#define TIMER_START(timer) \
    timer##_start_time = std::chrono::high_resolution_clock::now();
#define TIMER_FINISH(timer) \
    timer##_duration = std::chrono::high_resolution_clock::now() - timer##_start_time; \
    unsigned int timer##_res = std::chrono::duration_cast<std::chrono::microseconds>(timer##_duration).count(); \
    std::cout << TIMER_NAME(timer) << ": " << (timer##_res / 1000.0) << " ms" << std::endl;
#else
#include <iostream>
#include <QElapsedTimer>
#define TIMER_NAME(timer) #timer
#define TIMER_INIT(timer) QElapsedTimer timer;
#define TIMER_START(timer) timer.start();
#define TIMER_FINISH(timer) std::cout << TIMER_NAME(timer) << ": " << (timer.nsecsElapsed() / 1000000.0) << std::endl;
#endif
#else
#define TIMER_INIT(timer)
#define TIMER_START(timer)
#define TIMER_FINISH(timer)
#endif

#ifdef DETECTOR_DEBUG
#define DEBUG(x) x;
#else
#define DEBUG(x)
#endif

#define WARPED_SIZE 400

//=================================================================================================

static bool ratioCheck(float a1, float a2, float ratio = 0.3) {
    if(a1 > a2 && a2 / a1 > ratio) return true;
    else if(a1 / a2 > ratio) return true;
    return false;
}

static bool compareSize(const cv::Rect &r1, const cv::Rect &r2, float ratio = 0.3) {
    return ratioCheck(r1.width, r2.width, ratio) && ratioCheck(r1.height, r2.height, ratio);
}

static bool compareMinDistance(const cv::Rect &r1, const cv::Rect &r2, float ratio = 1.0) {
    return (abs(r1.x - r2.x) > ratio * (r1.width + r2.width) / 2.0
            || abs(r1.y - r2.y) > ratio * (r1.height + r2.height) / 2.0);

}

static bool compareMaxDistance(const cv::Rect &r1, const cv::Rect &r2, float ratio = 3.0) {
    int mw = (r1.width + r2.width) / 2;
    int mh = (r1.height + r2.height) / 2;
    return (abs(r1.x - r2.x) < ratio * mw && abs(r1.y - r2.y) < ratio * mh);
}

static unsigned int rectDistance(const cv::Rect &r1, const cv::Rect &r2) {
    cv::Point2i p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
    cv::Point2i p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

static float sizeRatio(const cv::Rect &r1, const cv::Rect &r2) {
    float a1 = r1.area();
    float a2 = r2.area();
    if(a1 == 0 || a2 == 0) return 0;
    if(a1 > a2) return a1 / a2;
    return a2 / a1;
}

static bool lineIntersects(cv::Point2i p11, cv::Point2i p12, cv::Point2i p21, cv::Point2i p22) {
    int s1 = (p21.y - p11.y) * (p12.x - p11.x) - (p21.x - p11.x) * (p12.y - p11.y) < 0 ? -1 : 1;
    int s2 = (p22.y - p11.y) * (p12.x - p11.x) - (p22.x - p11.x) * (p12.y - p11.y) < 0 ? -1 : 1;
    return s1 * s2 == -1;
}

//=================================================================================================

LabelDescriptor::LabelDescriptor(const cv::Rect &r1, const cv::Rect &r2, const cv::Rect &r3, const cv::Rect &rr) : ap(rr), id(-1), lifetime(0) {
    fips.push_back(r1);
    fips.push_back(r2);
    fips.push_back(r3);

    float maxd = 0;
    size_t maxi = 0;
    for(size_t i = 0; i < fips.size(); ++i) {
        float d = rectDistance(fips[i], fips[(i + 1) % 3]);
        if(d > maxd) {
            maxd = d;
            maxi = i;
        }
    }

    cv::Rect tl, tr, bl;
    tl = fips[(maxi + 2) % 3];
    if(fips[maxi].x < fips[(maxi + 1) % 3].x) {
        bl = fips[maxi];
        tr = fips[(maxi + 1) % 3];
    } else {
        bl = fips[(maxi + 1) % 3];
        tr = fips[maxi];
    }

    fips.clear();
    fips.push_back(tl);
    fips.push_back(tr);
    fips.push_back(bl);

    br = tl | tr | bl;
    eps = abs(rectDistance(tr, bl) - rectDistance(tl, tr) - rectDistance(tl, bl));
}

bool LabelDescriptor::isVisible() const {
    return lifetime == 0;
}

bool LabelDescriptor::operator <(const LabelDescriptor &other) const {
    return eps < other.eps;
}

struct PointComparator {
    bool operator ()(const cv::Point2i &p1, const cv::Point2i &p2) const {
        if(p1.x != p2.x) return p1.x < p2.x;
        else return p1.y < p2.y;
    }
};

//=================================================================================================

BRDetector::BRDetector(int cannyThreshold, float colorRatio, int satThreshold, int maxLifetime, int distThreshold)
    : objectCount(0), cannyThreshold(cannyThreshold), colorRatio(colorRatio), maxLifetime(maxLifetime), distThreshold(distThreshold), satThreshold(satThreshold),
      cornerApertureSize(3), cornerBlockSize(9), cornerK(4), cornerThreshold(175), focalLength(0) {

    objectPoints2D.push_back(cv::Point2f(0, 0));
    objectPoints2D.push_back(cv::Point2f(WARPED_SIZE, 0));
    objectPoints2D.push_back(cv::Point2f(0, WARPED_SIZE));
    objectPoints2D.push_back(cv::Point2f(WARPED_SIZE, WARPED_SIZE));

    objectPoints3D.push_back(cv::Point3f(-0.5f, -0.5f, 0));
    objectPoints3D.push_back(cv::Point3f(+0.5f, -0.5f, 0));
    objectPoints3D.push_back(cv::Point3f(-0.5f, +0.5f, 0));
    objectPoints3D.push_back(cv::Point3f(+0.5f, +0.5f, 0));
    
#ifdef WITH_GPU
    gpuWorker = 0;
#endif
}

BRDetector::~BRDetector() {
#ifdef WITH_GPU
    if(gpuWorker) delete gpuWorker;
#endif
}

void BRDetector::setCannyThreshold(int val) {
    cannyThreshold = val;
}

void BRDetector::setColorRatio(float val) {
    colorRatio = val;
}

void BRDetector::setMaxLifetime(int val) {
    maxLifetime = val;
}

void BRDetector::setDistThreshold(int val) {
    distThreshold = val;
}

void BRDetector::setSaturationThreshold(int val) {
    satThreshold = val;
}

const std::vector<LabelDescriptor> &BRDetector::getObjects() const {
    return objects;
}

void BRDetector::setCornerDetectorParams(int asize, int bsize, int k, int threshold) {
    cornerApertureSize = asize;
    cornerBlockSize = bsize;
    cornerK = k;
    cornerThreshold = threshold;
}

void BRDetector::setCameraParams(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, float focalLength, float realSize) {
    this->cameraMatrix = cameraMatrix;
    this->distCoeffs = distCoeffs;
    this->focalLength = focalLength;
    this->realSize = realSize;
}

//=================================================================================================

#define NOT_INTERSECTS(r1, r2) (((r1) & (r2)).area() == 0)
static std::vector<cv::Rect>::const_iterator findAlignmentMarker(const LabelDescriptor &d, const std::vector<cv::Rect> &rects) {
    cv::Rect br = d.fips[0] | d.fips[1] | d.fips[2];
    int brSize = br.area();
    cv::Rect mr = cv::Rect(0, 0, (d.fips[0].width + d.fips[1].width + d.fips[2].width) / 3, (d.fips[0].height + d.fips[1].height + d.fips[2].height) / 3);

    float sizeDist = 0.0;
    std::vector<cv::Rect>::const_iterator minR = rects.end();
    for(std::vector<cv::Rect>::const_iterator r = rects.begin(); r != rects.end(); ++r) {
        if(r->x > br.x && r->x < br.x + br.width && r->y > br.y && r->y < br.y + br.height
           && compareSize(*r, mr, 0.5)
           && lineIntersects(d.fips[1].tl(), d.fips[2].tl(), d.fips[0].tl(), r->tl())
           && NOT_INTERSECTS(*r, d.fips[0]) && NOT_INTERSECTS(*r, d.fips[1]) && NOT_INTERSECTS(*r, d.fips[2])

        ) {
            if(sizeDist < 1E-6 || abs(r->area() - brSize) < sizeDist) {
                sizeDist = abs(r->area() - brSize);
                minR = r;
            }
        }
    }
    return minR;
}
#undef NOT_INTERSECTS

/*
static void removeNestedRects(std::vector<cv::Rect> &rects) {
    if(rects.empty()) return;
    for(std::vector<cv::Rect>::iterator r1 = rects.begin(); r1 != rects.end() - 1;) {
        bool removed = false;
        for(std::vector<cv::Rect>::iterator r2 = r1 + 1; r2 != rects.end();) {
            unsigned int a = (*r1 | *r2).area();
            if(a == r1->area()) {
                r2 = rects.erase(r2);
            } else if(a == r2->area()) {
                r1 = rects.erase(r1);
                removed = true;
                break;
            } else ++r2;
        }
        if(!removed) ++r1;
    }
}
*/

static void mergeRects(std::vector<cv::Rect> &rects) {
    if(rects.size() < 2) return;
    for(std::vector<cv::Rect>::iterator r1 = rects.begin(); r1 != rects.end();) {
        bool merged = false;
        for(std::vector<cv::Rect>::iterator r2 = r1 + 1; r2 != rects.end(); ++r2) {
            if((*r1 & *r2).area() != 0) {
                *r2 = *r1 | *r2;
                r1 = rects.erase(r1);
                merged = true;
                break;
            }
        }
        if(!merged) ++r1;
    }
}

//=================================================================================================

void BRDetector::mergeRects(std::vector<cv::Rect> &rects1, std::vector<cv::Rect> &rects2) {
    for(std::vector<cv::Rect>::iterator r1 = rects1.begin(); r1 != rects1.end();) {
        bool found = false;
        for(std::vector<cv::Rect>::iterator r2 = rects2.begin(); r2 != rects2.end(); ++r2) {
            if((*r1 & *r2).area() > 0.6 * std::max(r1->area(), r2->area())) {
                rects2.erase(r2);
                found = true;
                break;
            }
        }
        if(found) ++r1;
        else r1 = rects1.erase(r1);
    }
}

void BRDetector::findRects(cv::Mat &frame, std::vector<cv::Rect> &blueRects, std::vector<cv::Rect> &redRects, bool storeEdges) {
    TIMER_INIT(opencv_channel_split);
    TIMER_START(opencv_channel_split);
    cv::Mat edges;
    std::vector<cv::Mat> bgrChannels;
    cv::split(frame, bgrChannels);
    TIMER_FINISH(opencv_channel_split);

#ifndef WITH_GPU
    TIMER_INIT(opencv_blur);
    TIMER_INIT(opencv_hsv);
    TIMER_INIT(opencv_canny);

    cv::Mat hsvFrame;

    TIMER_START(opencv_blur);
    cv::GaussianBlur(frame, frame, cv::Size(9, 9), 3);
    TIMER_FINISH(opencv_blur);

    TIMER_START(opencv_hsv);
    cv::cvtColor(frame, hsvFrame, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvFrame, hsvChannels);

    if(satThreshold) cv::threshold(hsvChannels[1], hsvChannels[1], satThreshold, 255, cv::THRESH_TOZERO);
    TIMER_FINISH(opencv_hsv);

    TIMER_START(opencv_canny);
    cv::Canny(hsvChannels[1], edges, cannyThreshold, cannyThreshold * 1.5);
    TIMER_FINISH(opencv_canny);
    DEBUG(hsvChannels[1].copyTo(this->satOut));
#else
    assert(gpuWorker != 0);
    gpuWorker->setImage(frame);
    gpuWorker->paintGL();
    cv::cvtColor(gpuWorker->renderOutput, edges, CV_BGR2GRAY);
#endif
    if(storeEdges) {
        edges.copyTo(edgesOut);
//        static cv::Mat e = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(3, 3));
//        cv::dilate(edgesOut, edgesOut, e);
    }

    TIMER_INIT(opencv_contours);
    TIMER_INIT(detector_select_fips);

    std::vector<std::vector<cv::Point2i> > contours;
    TIMER_START(opencv_contours);
    cv::findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    TIMER_FINISH(opencv_contours);

    TIMER_START(detector_select_fips);
    for(size_t i = 0; i < contours.size(); ++i) {
//        std::vector<cv::Point2i> poly;
//        cv::approxPolyDP(contours[i], poly, 3, true);
//        cv::Rect r = cv::boundingRect(poly);
        cv::Rect r = cv::boundingRect(contours[i]);

        if(r.width < 5 || r.height < 5) continue;
//        if(r.x + r.width > frame.cols || r.y + r.height > frame.rows) continue;
        unsigned int vred = cv::sum(bgrChannels[2](r))[0];
        unsigned int vgreen = cv::sum(bgrChannels[1](r))[0];
        unsigned int vblue = cv::sum(bgrChannels[0](r))[0];
        if(vblue > colorRatio * vred && vblue > colorRatio * vgreen) {
            blueRects.push_back(r);
        } else if(vred > colorRatio * vblue && vred > colorRatio * vgreen) {
            redRects.push_back(r);
        }
    }
    TIMER_FINISH(detector_select_fips);
    //    mergeRects(blueRects);
    //    mergeRects(redRects);
}

std::vector<LabelDescriptor> BRDetector::findLabels(const std::vector<cv::Rect> &blueRects, const std::vector<cv::Rect> &redRects) const {
    TIMER_INIT(detector_build_graph);
    TIMER_INIT(detector_find_cycles);

    // build fip graph
    TIMER_START(detector_build_graph);
    std::vector<std::vector<char> > adj;
    int vc = 1;
    for(std::vector<cv::Rect>::const_iterator v1 = blueRects.begin(); v1 != blueRects.end(); ++v1, ++vc) {
        adj.push_back(std::vector<char>(vc, 0));
        for(std::vector<cv::Rect>::const_iterator v2 = v1 + 1; v2 != blueRects.end(); ++v2) {
            if(compareSize(*v1, *v2, 0.25)
                    && compareMaxDistance(*v1, *v2, 3.0)
                    && compareMinDistance(*v1, *v2, 0.5)
                    ) {
                //cv::line(cframe, RECT_CENTER(*v1), RECT_CENTER(*v2), cv::Scalar(255, 255, 0), 2);
                adj.back().push_back(1);
            } else {
                adj.back().push_back(0);
            }
        }
    }
    TIMER_FINISH(detector_build_graph);

    TIMER_START(detector_find_cycles);
    // find 3-cycles in fip graph
    std::vector<LabelDescriptor> lds;
    for(size_t i = 0; i < blueRects.size(); ++i) {
        for(size_t j = i + 1; j < blueRects.size(); ++j) {
            if(!adj[i][j]) continue;
            for(size_t k = j + 1; k < blueRects.size(); ++k) {
                if(adj[j][k] && adj[i][k]) {
                    LabelDescriptor d(blueRects[i], blueRects[j], blueRects[k]);
                    std::vector<cv::Rect>::const_iterator rr = findAlignmentMarker(d, redRects);
                    if(rr != redRects.end()) {
                        d.ap = *rr;
                        lds.push_back(d);
                    }
                }
            }
        }
    }
    TIMER_FINISH(detector_find_cycles);

    if(lds.size() > 1) {
        TIMER_INIT(detector_find_groups);
        TIMER_START(detector_find_groups);
        // group descriptors with intersections
        std::vector<std::vector<LabelDescriptor> > groups;
        int g = 0;
        std::map<cv::Point2i, int, PointComparator> groupMap;
        for(std::vector<LabelDescriptor>::iterator d = lds.begin(); d != lds.end(); ++d) {
            int cgroup = -1;
            for(int i = 0; i < 3; ++i) {
                std::map<cv::Point2i, int>::iterator f = groupMap.find(d->fips[i].tl());
                if(f != groupMap.end()) {
                    cgroup = f->second;
                    break;
                }
            }
            int addgroup = cgroup == -1 ? g++ : cgroup;
            for(int i = 0; i < 3; ++i) groupMap[d->fips[i].tl()] = addgroup;
            if(cgroup == -1) {
                groups.push_back(std::vector<LabelDescriptor>());
                groups.back().push_back(*d);
            } else {
                groups[cgroup].push_back(*d);
            }
        }

        // sort descriptors in groups by max rightness and leave only best matches
        lds.clear();
        for(std::vector<std::vector<LabelDescriptor> >::iterator g = groups.begin(); g != groups.end(); ++g) {
            std::sort(g->begin(), g->end());
            lds.push_back(g->at(0));
        }
        TIMER_FINISH(detector_find_groups);
    }

    return lds;
}

//=================================================================================================

#define PT_DIST(p1, p2) ((p1.x - p2->x) * (p1.x - p2->x) + (p1.y - p2->y) * (p1.y - p2->y))
cv::Point BRDetector::findCorner(const cv::Mat &img) const {
    cv::Mat dst, dst_norm;

    cv::cornerHarris(img, dst, cornerBlockSize, cornerApertureSize % 2 == 1 ? cornerApertureSize : cornerApertureSize + 1, (float)cornerK / 100.0, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    std::vector<cv::Point> corners;
    for(int j = 0; j < dst_norm.rows; j++) {
        for(int i = 0; i < dst_norm.cols; i++) {
            if(dst_norm.at<float>(j, i) > cornerThreshold) corners.push_back(cv::Point(i, j));
        }
    }

    if(corners.empty()) return cv::Point(-1, -1);

    cv::Point sp = corners[0];
    if(corners.size() > 1) {
        int maxDist = 0;
        for(std::vector<cv::Point>::iterator i = corners.begin() + 1; i != corners.end(); ++i) {
            int dist = PT_DIST(sp, i);
            if(dist > maxDist) maxDist = dist;
        }

        for(std::vector<cv::Point>::iterator i = corners.begin() + 1; i != corners.end(); ++i) {
            if(PT_DIST(sp, i) < maxDist / 2) sp = cv::Point((sp.x + i->x) / 2, (sp.y + i->y) / 2);
        }
    }
    return sp;
}
#undef PT_DIST

#define CORNER_BR   0
#define CORNER_BL   1
#define CORNER_TR   2
#define CORNER_TL   3

cv::Point BRDetector::findCorner(const cv::Mat &img, const cv::Point &offset, int corner) const {
    int iw = img.cols / 2, ih = img.rows / 2;

    cv::Point c(-1, -1);
    switch(corner) {
    case CORNER_TL:
        c = findCorner(img(cv::Rect(0, 0, iw, ih)));
        if(c.x != -1) return c + offset;
        break;
    case CORNER_BR:
        c = findCorner(img(cv::Rect(img.cols - iw, img.rows - ih, iw, ih)));
        if(c.x != -1) return c + offset + cv::Point(img.cols - iw, img.rows - ih);
        break;
    case CORNER_TR:
        c = findCorner(img(cv::Rect(img.cols - iw, 0, iw, ih)));
        if(c.x != -1) return c + offset + cv::Point(img.cols - iw, 0);
        break;
    case CORNER_BL:
        c = findCorner(img(cv::Rect(0, img.rows - ih, iw, ih)));
        if(c.x != -1) return c + offset + cv::Point(0, img.rows - ih);
        break;
    default:
        break;
    }
    return c;
}

static cv::Rect adjustRect(const cv::Rect &r, int v, const cv::Mat &img) {
    cv::Rect rect(r.x - v, r.y - v, r.width + v, r.height + v);
    if(rect.x < 0) rect.x = 0;
    if(rect.x + rect.width > img.cols) rect.width = img.cols - rect.x;
    if(rect.y < 0) rect.y = 0;
    if(rect.y + rect.height > img.rows) rect.height = img.rows - rect.y;
    return rect;
}

bool BRDetector::findQRCorners(const cv::Mat &edges, LabelDescriptor &ld) const {
    ld.corners.clear();

    for(int i = 0; i < 3; ++i) {
        cv::Rect r = adjustRect(ld.fips[i], 2, edges);
        cv::Point c = findCorner(edges(r), r.tl(), i);
        if(c.x == -1) return false;
        ld.corners.push_back(c);
    }

    cv::Rect r = adjustRect(ld.ap, 2, edges);
    cv::Point c = findCorner(edges(r), r.tl(), CORNER_TL);
    if(c.x == -1) return false;
    ld.corners.push_back(c);

    static cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
    cv::cornerSubPix(edges, ld.corners, cv::Size(9, 9), cv::Size(-1, -1), termCriteria);

    return true;
}

//=================================================================================================

bool BRDetector::readLabel(const cv::Mat &frame, int objID) {
    for(std::vector<LabelDescriptor>::iterator d = objects.begin(); d != objects.end(); ++d) {
        if(d->id == objID) return readLabel(frame, *d);
    }
    return false;
}

bool BRDetector::readLabel(const cv::Mat &frame, const cv::Mat &edges, LabelDescriptor &ld, cv::Mat *warpedOut) const {
    if(!findQRCorners(edges, ld)) return false;
    return readLabel(frame, ld, warpedOut);
}

bool BRDetector::readLabel(const cv::Mat &frame, LabelDescriptor &ld, cv::Mat *warpedOut) const {
    if(ld.corners.size() != 4) return false;

    cv::Mat H = cv::getPerspectiveTransform(ld.corners, objectPoints2D);
    cv::Mat warped(WARPED_SIZE, WARPED_SIZE, frame.type());
    cv::warpPerspective(frame, warped, H, cv::Size(WARPED_SIZE, WARPED_SIZE));
    cv::cvtColor(warped, warped, CV_BGR2GRAY);
    if(warpedOut) warped.copyTo(*warpedOut);

    bool detected = false;
#ifdef WITH_ZBAR
    zbar::ImageScanner scanner;
    scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
    scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);

    zbar::Image zimg(warped.cols, warped.rows, "GREY", warped.data, warped.rows * warped.cols * warped.elemSize());
    if(scanner.scan(zimg) > 0) {
        const zbar::SymbolSet &syms = scanner.get_results();
        ld.data = syms.symbol_begin()->get_data();
        detected = true;
    }
    zimg.set_data(NULL, 0);
#endif
    return detected;
}

//=================================================================================================
/*
 position:
  - source 1: https://github.com/MasteringOpenCV/code/blob/master/Chapter2_iPhoneAR/Example_MarkerBasedAR/Example_MarkerBasedAR/MarkerDetector.cpp
  - source 2: http://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-roational-matrix
  - alternative: http://stackoverflow.com/questions/8927771/computing-camera-pose-with-homography-matrix-based-on-4-coplanar-points
 distance:
  - source: http://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv
  - alternative: http://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image
*/
bool BRDetector::estimatePosition(LabelDescriptor &ld) const {
    if(ld.corners.size() != 4) return false;

    cv::Mat Rvec, Tvec, raux, taux;
    cv::solvePnP(objectPoints3D, ld.corners, cameraMatrix, distCoeffs, raux, taux);
    raux.convertTo(Rvec, CV_32F);
    taux.convertTo(Tvec, CV_32F);
    cv::Mat_<float> rotMat(3,3);
    cv::Rodrigues(Rvec, rotMat);
    cv::transpose(rotMat, rotMat);

    float r32 = rotMat.at<float>(2, 1);
    float r33 = rotMat.at<float>(2, 2);
    ld.xAngle = atan2(r32, r33);
    ld.yAngle = atan2(-rotMat.at<float>(2, 0), sqrt(r32 * r32 + r33 * r33));
    ld.zAngle = atan2(rotMat.at<float>(1, 0), rotMat.at<float>(0, 0));

    ld.distance = realSize * focalLength * 2.0 / (float)(ld.br.height + ld.br.width);

    return true;
}

//=================================================================================================

void BRDetector::trackObjects(const std::vector<LabelDescriptor> &curLabels) {
    // if no labels detected in current frame - just increase lifetime of all objects
    if(curLabels.empty()) {
        for(std::vector<LabelDescriptor>::iterator d = objects.begin(); d != objects.end();) {
            if(++d->lifetime == maxLifetime) d = objects.erase(d);
            else ++d;
        }
        return;
    }

    // if currently no objects detected - save all labels fro current frame
    if(objects.empty()) {
        objects = curLabels;
        for(size_t i = 0; i < objects.size(); ++i)
            objects[i].id = objectCount + i;
        objectCount += objects.size();
        return;
    }

    // compute distance matrix between old and new objects
    int **distMatrix = new int*[objects.size()];
    for(size_t i = 0; i < objects.size(); ++i) {
        distMatrix[i] = new int[curLabels.size()];
        for(size_t j = 0; j < curLabels.size(); ++j) {
            unsigned int dist = rectDistance(objects[i].br, curLabels[j].br);
            if(dist > distThreshold || !compareSize(objects[i].br, curLabels[j].br, 0.5)) distMatrix[i][j] = 2 * distThreshold;
            else distMatrix[i][j] = dist;
        }
    }

    // solve assignment problem
    hungarian_problem_t p;
    hungarian_init(&p, distMatrix, objects.size(), curLabels.size(), HUNGARIAN_MODE_MINIMIZE_COST);
    hungarian_solve(&p);

    // update position of detected objects
    std::map<int, bool> assignedLabels;
    for(size_t i = 0; i < objects.size(); ++i) {
        bool found = false;
        for(size_t j = 0; j < curLabels.size(); ++j) {
            if(p.assignment[i][j] == 0) continue;
            if(p.cost[i][j] > distThreshold) break; // bad match found

            assignedLabels[j] = true;
            int objID = objects[i].id;
            objects[i] = curLabels[j];
            objects[i].id = objID;
            found = true;
            break;
        }
        if(!found) ++objects[i].lifetime;
    }

    // remove obsolete objects
    size_t oldObjectsSize = objects.size();
    for(std::vector<LabelDescriptor>::iterator d = objects.begin(); d != objects.end();) {
        if(d->lifetime == maxLifetime) d = objects.erase(d);
        else ++d;
    }

    // add new objects
    if(curLabels.size() > oldObjectsSize) {
        for(size_t i = 0; i < curLabels.size(); ++i) {
            if(assignedLabels.find(i) == assignedLabels.end()) {
                objects.push_back(curLabels[i]);
                objects.back().id = objectCount;
                ++objectCount;
            }
        }
    }

    hungarian_free(&p);
    for(size_t i = 0; i < objects.size(); ++i) delete[] distMatrix[i];
    delete[] distMatrix;
}

//=================================================================================================

std::vector<LabelDescriptor> BRDetector::detect(cv::Mat &frame, std::vector<cv::Rect> *blueRectsOut, std::vector<cv::Rect> *redRectsOut, bool storeEdges) {
    TIMER_INIT(total_time);
    TIMER_INIT(detector_find_rects);
    TIMER_START(total_time);
    TIMER_START(detector_find_rects);
    std::vector<cv::Rect> blueRects, redRects;
    findRects(frame, blueRects, redRects, storeEdges);
    TIMER_FINISH(detector_find_rects);

    if(blueRectsOut) *blueRectsOut = blueRects;
    if(redRectsOut) *redRectsOut = redRects;

#ifdef PERF_TEST
    TIMER_INIT(detector_find_labels);
    TIMER_START(detector_find_labels);
    std::vector<LabelDescriptor> res = findLabels(blueRects, redRects);
    TIMER_FINISH(detector_find_labels);
    TIMER_FINISH(total_time);
    return res;
#else
    return findLabels(blueRects, redRects);
#endif
}

std::vector<LabelDescriptor> BRDetector::detect2(cv::Mat &frame1, cv::Mat &frame2) {
    std::vector<cv::Rect> blueRects1, redRects1, blueRects2, redRects2;

    findRects(frame1, blueRects1, redRects1);
    findRects(frame2, blueRects2, redRects2);
    mergeRects(blueRects1, blueRects2);
    mergeRects(redRects1, redRects2);

    return findLabels(blueRects1, blueRects2);
}

#ifdef WITH_GPU
void BRDetector::createGPUWorker(int w, int h, void *data) {
    if(gpuWorker) return;
    gpuWorker = new GPUWorker(w, h, data);
}

bool BRDetector::initGPU(const std::string &blur, const std::string &hsv, const std::string &sobel, const std::string &canny) {
    if(!gpuWorker) return false;
#ifdef GPU_DIRECT_INIT
    gpuWorker->initializeGL();
#endif
    return gpuWorker->loadShaders(blur, hsv, sobel, canny);
}
#endif
