#include "gpu_worker_win.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <iostream>

//=====================================================================================================================

#ifdef PERF_TEST
#include <iostream>
#include <QElapsedTimer>
#define TIMER_NAME(timer) #timer
#define TIMER_INIT(timer) QElapsedTimer timer;
#define TIMER_START(timer) timer.start();
#define TIMER_FINISH(timer) std::cout << TIMER_NAME(timer) << ": " << (timer.nsecsElapsed() / 1000000.0) << std::endl;
#else
#define TIMER_INIT(timer)
#define TIMER_START(timer)
#define TIMER_FINISH(timer)
#endif

#define GET_STATUS(type, id, status, res) {             \
    res = GL_FALSE;                                     \
    glGet##type##iv(id, status, &res);                  \
    if(!res) {                                          \
        GLint ls = 0;                                   \
        glGet##type##iv(id, GL_INFO_LOG_LENGTH, &ls);   \
        if(ls > 0) {                                    \
            char *msg = new char[ls];                   \
            glGet##type##InfoLog(id, ls, NULL, msg);    \
            std::cout << msg << std::endl;              \
            delete[] msg;                               \
        } else std::cout << "unlnown error" << std::endl;\
    }                                                   \
}

#define INIT_TEX(id, w, h, f) \
    glGenTextures(1, &id);\
    glBindTexture(GL_TEXTURE_2D, id);\
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);\
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, f);\
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, f);\
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);\
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);\

#define DRAW_QUAD(p) \
    glEnableVertexAttribArray(p->vs); \
    glBindBuffer(GL_ARRAY_BUFFER, state.rqID); \
    glVertexAttribPointer(p->vs, 3, GL_FLOAT, GL_FALSE, 0, 0); \
    glDrawArrays(GL_TRIANGLES, 0, 6); \
    glDisableVertexAttribArray(p->vs); \
    glFinish(); \

#define check() assert(glGetError() == 0)

//-------------------------------------------------------------------------------------------------

#define NUM_PROGRAMS    4
#define BLUR_PROGRAM    0
#define HSV_PROGRAM     1
#define SOBEL_PROGRAM   2
#define CANNY_PROGRAM   3

GPUWorker::GPUWorker(int w, int h, void *parent) : QOpenGLWidget((QWidget*)parent), imgWidth(w), imgHeight(h) {
    renderOutput = cv::Mat(imgHeight, imgWidth, CV_8UC3);
}

GPUWorker::~GPUWorker() {
    makeCurrent();
    glDeleteFramebuffers(2, state.fb);
    glDeleteBuffers(1, &state.rqID);
    glDeleteTextures(3, state.tex);
    glDeleteShader(state.vsID);
    for(size_t i = 0; i < programs.size(); ++i) {
        glDeleteProgram(programs[i]->id);
        delete programs[i];
    }
    doneCurrent();
}

//-------------------------------------------------------------------------------------------------

bool GPUWorker::loadShaders(const std::string &blur, const std::string &hsv, const std::string &sobel, const std::string &canny) {
    makeCurrent();
    if(!addFragmentShader(blur)) return false;
    if(!addFragmentShader(hsv)) return false;
    if(!addFragmentShader(sobel)) return false;
    if(!addFragmentShader(canny)) return false;

    BlurProgram *bp = new BlurProgram(programs[BLUR_PROGRAM]);
    bp->radius = glGetUniformLocation(bp->id, "radius");
    bp->hor = glGetUniformLocation(bp->id, "horizontal");
    glUseProgram(bp->id);
    glUniform1f(bp->radius, 1.0);

    HSVProgram *hp = new HSVProgram(programs[HSV_PROGRAM]);
    hp->threshold = glGetUniformLocation(hp->id, "threshold");
    glUseProgram(hp->id);
    glUniform1f(hp->threshold, 0.392157); // 100 / 255

    SobelProgram *sp = new SobelProgram(programs[SOBEL_PROGRAM]);
    sp->hor = glGetUniformLocation(sp->id, "horizontal");

    CannyProgram *cp = new CannyProgram(programs[CANNY_PROGRAM]);

    for(size_t i = 0; i < programs.size(); ++i) delete programs[i];
    programs[BLUR_PROGRAM] = bp;
    programs[HSV_PROGRAM] = hp;
    programs[SOBEL_PROGRAM] = sp;
    programs[CANNY_PROGRAM] = cp;
    doneCurrent();
    return true;
}

//-------------------------------------------------------------------------------------------------

void GPUWorker::setImage(const cv::Mat &img) {
    makeCurrent();
    TIMER_INIT(gpu_set_img);
    TIMER_START(gpu_set_img);
    glBindTexture(GL_TEXTURE_2D, state.tex[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    TIMER_FINISH(gpu_set_img);
    doneCurrent();
}

//-------------------------------------------------------------------------------------------------

void GPUWorker::initializeGL() {
    initializeOpenGLFunctions();
    // OpenGL initialized
    const char *glsl_ver = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    if(glGetError() != GL_NO_ERROR) std::cout << "unble to get GLSL version: " << glGetError()  << std::endl;
    else std::cout << glGetString(GL_VERSION) << " | " << glsl_ver << std::endl;

    //create passtrough vertex shader
    state.vsID = glCreateShader(GL_VERTEX_SHADER);
    const char *dvsCode =
            "attribute vec3 vertexPosition;                 \n"
            "varying vec2 UV;                               \n"
            "void main() {                                  \n"
            "  gl_Position = vec4(vertexPosition, 1);       \n"
            "  UV = (vertexPosition.xy + vec2(1, 1)) / 2.0; \n"
            "}                                              \n";
    glShaderSource(state.vsID, 1, &dvsCode, NULL);
    glCompileShader(state.vsID);

    GLint result = GL_FALSE;
    GET_STATUS(Shader, state.vsID, GL_COMPILE_STATUS, result);
    assert(result != GL_FALSE);

    //create render quad
    static const GLfloat rq_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };

    glGenBuffers(1, &state.rqID);
    glBindBuffer(GL_ARRAY_BUFFER, state.rqID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rq_data), rq_data, GL_STATIC_DRAW);

    INIT_TEX(state.tex[0], imgWidth, imgHeight, GL_NEAREST);
    INIT_TEX(state.tex[1], imgWidth, imgHeight, GL_NEAREST);
    INIT_TEX(state.tex[2], imgWidth, imgHeight, GL_NEAREST);

    //create framebuffer for non-screen render targets
    glGenFramebuffers(2, state.fb);
    glBindFramebuffer(GL_FRAMEBUFFER, state.fb[0]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, state.tex[1], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, state.tex[2], 0);

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    std::cout << "OpenGL configured" << std::endl;
}

//-------------------------------------------------------------------------------------------------

void GPUWorker::paintGL() {
    makeCurrent();
    if(programs.size() < NUM_PROGRAMS) return;
    GLenum drawBuffers[2][1] = {{GL_COLOR_ATTACHMENT0}, {GL_COLOR_ATTACHMENT1}};

    TIMER_INIT(gpu_blur);
    TIMER_INIT(gpu_hsv);
    TIMER_INIT(gpu_canny);

    glBindFramebuffer(GL_FRAMEBUFFER, state.fb[0]);
    glDrawBuffers(1, drawBuffers[0]);
    glViewport(0, 0, imgWidth, imgHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Blur | tex[0] -> tex[1] -> tex[2]
    TIMER_START(gpu_blur);
    BlurProgram *bp = static_cast<BlurProgram*>(programs[BLUR_PROGRAM]);
    glUseProgram(bp->id);
    glUniform1i(bp->hor, GL_TRUE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, state.tex[0]);
    glUniform1i(bp->tex, 0);
    DRAW_QUAD(bp);
    glUniform1i(bp->hor, GL_FALSE);
    glDrawBuffers(1, drawBuffers[1]);
    glBindTexture(GL_TEXTURE_2D, state.tex[1]);
    DRAW_QUAD(bp);
    TIMER_FINISH(gpu_blur);

    // HSV | tex[2] -> tex[2]
    TIMER_START(gpu_hsv);
    HSVProgram *hp = static_cast<HSVProgram*>(programs[HSV_PROGRAM]);
    glUseProgram(hp->id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, state.tex[2]);
    glUniform1i(hp->tex, 0);
    DRAW_QUAD(hp);
    TIMER_FINISH(gpu_hsv);

    // Sobel | tex[2] -> tex[1] -> tex[2]
    TIMER_START(gpu_canny);
    SobelProgram *sp = static_cast<SobelProgram*>(programs[SOBEL_PROGRAM]);

    TIMER_START(gpu_canny);
    glUseProgram(sp->id);
    glDrawBuffers(1, drawBuffers[0]);
    glUniform1i(sp->hor, GL_TRUE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, state.tex[2]);
    glUniform1i(sp->tex, 0);
    DRAW_QUAD(sp);

    glDrawBuffers(1, drawBuffers[1]);
    glUniform1i(sp->hor, GL_FALSE);
    glBindTexture(GL_TEXTURE_2D, state.tex[1]);
    DRAW_QUAD(sp);

    // Canny | tex[2] -> tex[1]
    CannyProgram *cp = static_cast<CannyProgram*>(programs[CANNY_PROGRAM]);
    glUseProgram(cp->id);
    glDrawBuffers(1, drawBuffers[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, state.tex[2]);
    glUniform1i(cp->tex, 0);
    DRAW_QUAD(cp);
    TIMER_FINISH(gpu_canny);

    // Read | tex[1] -> output
    TIMER_INIT(gpu_get_img);
    TIMER_START(gpu_get_img);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, imgWidth, imgHeight, GL_RGB, GL_UNSIGNED_BYTE, renderOutput.data);
    TIMER_FINISH(gpu_get_img);
    doneCurrent();
}

//-------------------------------------------------------------------------------------------------

bool GPUWorker::addFragmentShader(const std::string &path) {
    FILE *fin = fopen(path.c_str(), "rb");
    if(!fin) {
        std::cerr << "Unable to open file: " << path << std::endl;
        return false;
    }

    fseek(fin, 0, SEEK_END);
    size_t fsize = ftell(fin);
    if(fsize > 1024 * 1024) {
        fclose(fin);
        return false;
    }

    char *fdata = new char[fsize + 1];
    fseek(fin, 0, SEEK_SET);
    if(fread(fdata, sizeof(char), fsize, fin) != fsize) {
        std::cerr << "Unable to read file: " << path << std::endl;
        fclose(fin);
        delete[] fdata;
        return false;
    }
    fdata[fsize] = 0;
    fclose(fin);

    GLuint sid = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(sid, 1, (const GLchar**)&fdata, NULL);
    glCompileShader(sid);

    delete[] fdata;

    GLint result = GL_FALSE;
    GET_STATUS(Shader, sid, GL_COMPILE_STATUS, result);
    if(!result) return false;

    GLuint pid = glCreateProgram();
    glAttachShader(pid, state.vsID);
    glAttachShader(pid, sid);
    glLinkProgram(pid);
    glDeleteShader(sid);

    GET_STATUS(Program, pid, GL_LINK_STATUS, result);
    if(!result) return false;

    GLuint vp = glGetAttribLocation(pid, "vertexPosition");
    GLuint it = glGetUniformLocation(pid, "inputTexture");
    GLuint sz = glGetUniformLocation(pid, "size");
    programs.push_back(new Program(pid, vp, it, sz));

    glUseProgram(pid);
    glUniform2f(sz, imgWidth, imgHeight);
    return true;
}
