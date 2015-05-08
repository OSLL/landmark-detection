#ifndef GPU_WORKER_H
#define GPU_WORKER_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Compatibility>
#include <QList>

#include "opencv2/core/core.hpp"

class GPUWorker : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Compatibility {
    Q_OBJECT

public:
    cv::Mat renderOutput;

    GPUWorker(int w = 640, int h = 480, void *parent = 0);
    virtual ~GPUWorker();

    bool loadShaders(const std::string &blur, const std::string &hsv, const std::string &sobel, const std::string &canny);
    void setImage(const cv::Mat &img);
    void initializeGL();
    void paintGL();

private:
    bool addFragmentShader(const std::string &path);

    struct State {
       GLuint tex[3];
       GLuint vsID;  //default vertex shader
       GLuint fb[2]; //output framebuffer
       GLuint rqID;  //rendered quad
    };

    struct Program {
        Program(GLuint i = 0, GLuint v = 0, GLuint t = 0, GLuint s = 0) : id(i), vs(v), tex(t), size(s) {}
        GLuint id, vs, tex, size;
    };

    struct BlurProgram : public Program {
        BlurProgram(const Program *p) : Program(p->id, p->vs, p->tex, p->size) {}
        GLuint radius, hor;
    };

    struct HSVProgram : public Program {
        HSVProgram(const Program *p) : Program(p->id, p->vs, p->tex, p->size) {}
        GLuint threshold;
    };

    struct SobelProgram : public Program {
        SobelProgram(const Program *p) : Program(p->id, p->vs, p->tex, p->size) {}
        GLuint hor;
    };

    struct CannyProgram : public Program {
        CannyProgram(const Program *p) : Program(p->id, p->vs, p->tex, p->size) {}
        GLuint threshold;
    };

    State state;
    GLsizei imgWidth, imgHeight;
    QList<Program*> programs;
};

#endif
