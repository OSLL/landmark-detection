#ifndef BR_DETECTOR_H
#define BR_DETECTOR_H

#include <vector>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"

struct LabelDescriptor {
    LabelDescriptor(const cv::Rect &r1, const cv::Rect &r2, const cv::Rect &r3, const cv::Rect &rr = cv::Rect());

    bool isVisible() const;
    bool operator <(const LabelDescriptor &other) const;

    std::vector<cv::Rect> fips;
    std::vector<cv::Point2f> corners;
    cv::Rect ap, br;
    std::string data;
    int eps, id, lifetime;
    float xAngle, yAngle, zAngle, distance;
};

class BRDetector {
public:
    BRDetector(int cannyThreshold = 100, float colorRatio = 1.6, int satThreshold = 0, int maxLifetime = 24, int distThreshold = 100);
    ~BRDetector();

    void setCannyThreshold(int val);
    void setColorRatio(float val);
    void setMaxLifetime(int val);
    void setDistThreshold(int val);
    void setSaturationThreshold(int val);
    void setCornerDetectorParams(int asize, int bsize, int k, int threshold);
    void setCameraParams(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, float focalLength = 0, float realSize = 0);

    std::vector<LabelDescriptor> detect(cv::Mat &frame, std::vector<cv::Rect> *blueRectsOut = NULL, std::vector<cv::Rect> *redRectsOut = NULL, bool storeEdges = false);
    std::vector<LabelDescriptor> detect2(cv::Mat &frame1, cv::Mat &frame2);

    void trackObjects(const std::vector<LabelDescriptor> &curLabels);

    void findRects(cv::Mat &frame, std::vector<cv::Rect> &blueRects, std::vector<cv::Rect> &redRects, bool storeEdges = false);
    void mergeRects(std::vector<cv::Rect> &rects1, std::vector<cv::Rect> &rects2);
    std::vector<LabelDescriptor> findLabels(const std::vector<cv::Rect> &blueRects, const std::vector<cv::Rect> &redRects) const;
    bool findQRCorners(const cv::Mat &edges, LabelDescriptor &ld) const;

    bool readLabel(const cv::Mat &frame, const cv::Mat &edges, LabelDescriptor &ld, cv::Mat *warpedOut = NULL) const;
    bool readLabel(const cv::Mat &frame, LabelDescriptor &ld, cv::Mat *warpedOut = NULL) const;
    bool readLabel(const cv::Mat &frame, int objID);

    bool estimatePosition(LabelDescriptor &ld) const;

    const std::vector<LabelDescriptor> &getObjects() const;

    cv::Mat edgesOut;
#ifdef DETECTOR_DEBUG
    mutable cv::Mat satOut;
#endif

private:
    cv::Point findCorner(const cv::Mat &img) const;
    cv::Point findCorner(const cv::Mat &img, const cv::Point &offset, int corner) const;

    std::vector<LabelDescriptor> objects;
    unsigned int objectCount;
    int cannyThreshold, maxLifetime, distThreshold, satThreshold;
    float colorRatio;
    int cornerApertureSize, cornerBlockSize, cornerK, cornerThreshold;

    cv::Mat cameraMatrix, distCoeffs;
    // focalLength = pixelWidth * knownDistance / knownWidth
    float focalLength, realSize;
    std::vector<cv::Point2f> objectPoints2D;
    std::vector<cv::Point3f> objectPoints3D;
};

#endif
