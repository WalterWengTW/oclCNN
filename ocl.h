/* =========================================================*//*
 * OpenCL based Convolutional Neural Network Framework
 *
 * Author : Walter Weng
 * Date : 2022/08/03
 *
 */

#ifndef INCLUDE_OCL_H_
#define INCLUDE_OCL_H_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CL_TARGET_OPENCL_VERSION 200
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <vector>
#include <cstring>
#include <assert.h>


typedef float OCL_TYPE;
typedef cl_platform_id   OCL_PLATFORM;
typedef cl_device_id     OCL_DEVICE;
typedef cl_context       OCL_CONTEXT;
typedef cl_command_queue OCL_QUEUE;
typedef cl_program       OCL_PROGRAM;
typedef cl_kernel        OCL_KERNEL;
typedef cl_mem           OCL_MEM;
typedef cl_mem_flags     OCL_MFLAGS;
typedef cl_event         OCL_EVENT;

extern OCL_DEVICE  dDevice;
extern OCL_CONTEXT dContext;
extern OCL_QUEUE   dQueue;
extern OCL_PROGRAM dProgram;



typedef struct shape
{
    int N;
    int C;
    int H;
    int W;
}shape_t;


namespace ocl  /* ocl.cpp */
{
    std::vector<OCL_PLATFORM> getPlatformID();
    std::vector<OCL_DEVICE>   getDeviceID(OCL_PLATFORM p, cl_device_type type = CL_DEVICE_TYPE_GPU);
    OCL_CONTEXT createContext(OCL_PLATFORM p, OCL_DEVICE d);
    OCL_QUEUE createQueue(OCL_CONTEXT c, OCL_DEVICE d);
    OCL_PROGRAM createProgramWithSrc(OCL_CONTEXT c, std::vector<std::string> fileNames);
    OCL_KERNEL createKernel(OCL_PROGRAM p, const char* k);
    OCL_MEM createBuffer(OCL_CONTEXT c, OCL_MFLAGS f, size_t s, void *h = nullptr);
}

namespace ocl /* basic.cpp */
{
    void getPlatformName(OCL_PLATFORM p);
    void getDeviceName(OCL_DEVICE d);
    size_t getFileData(std::string &fileName, char* s);
    void buildProgram(OCL_PROGRAM p, OCL_DEVICE d, const char* options = "-O3 -cl-mad-enable");
    void getBuildInfo(OCL_PROGRAM p, OCL_DEVICE d);
    void setArg(OCL_KERNEL k, std::vector<void*> args, std::vector<size_t> argSizes);
    void writeBuffer(OCL_QUEUE q, OCL_MEM m, size_t s, void *ptr, OCL_EVENT *e = nullptr);
    void readBuffer(OCL_QUEUE q, OCL_MEM m, size_t s, void *ptr, OCL_EVENT *e = nullptr);
    void ndRangeKernel(OCL_QUEUE q, OCL_KERNEL k, size_t d, const size_t* gws, const size_t* lws, OCL_EVENT *e = nullptr);
    void getErrMsg(cl_int e);
    void getLocalSet(size_t gws[3], size_t lws[3]);
}


namespace ocl
{
    /* layer_basic.cpp */
    class layer
    {
    protected:
        std::string name;
        void* cMEM = nullptr;
        OCL_MEM gMEM;
        layer* prvLayer;
        shape_t shape;
        size_t s_;
        size_t nParam = 0;

    public:
        layer() {  }
        ~layer() { }
        virtual void run() = 0;
        virtual size_t setParam(OCL_TYPE* buffer) = 0;
        virtual void setKernel() = 0;
        void setCPUMem(void* m) { cMEM = m; }
        void* getCPUMem(void* m) { return cMEM; }

        void setGPUMem(OCL_MEM m) { gMEM = m; }
        OCL_MEM getGPUMem() { return gMEM; }

        void setName(std::string n) { name = n; }
        std::string getName() { return name; }

        void setPrvLayer(layer* p) { prvLayer = p; }
        layer* getPrvLayer() { return prvLayer; }

        void detach();
        shape_t getShape() { return shape; }
        void setShape(shape_t s) { shape = s; }

        size_t getNParam() { return nParam; }
        void setNParam(size_t n) { nParam = n; }
    };

    /* layer_input.cpp */
    class input : public layer
    {
    public:
        input(shape_t s);
        ~input() { }
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_conv2d.cpp */
    class conv2d : public layer
    {
    protected:
        OCL_MEM wMEM = nullptr;
        OCL_MEM bMEM = nullptr;
        OCL_MEM iMEM = nullptr;
        OCL_KERNEL kernel;
        bool useBias;
        int padSize;
        int stride;
        int kSize;
        int channel_in;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
        size_t weightSize;
        size_t biasSize;
    public:
        conv2d(layer* p, int u, int ks, int s = 1, int pad = 0, bool use_bias = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_pwconv2d.cpp */
    class pwconv2d : public layer
    {
    protected:
        OCL_MEM wMEM = nullptr;
        OCL_MEM bMEM = nullptr;
        OCL_MEM iMEM = nullptr;
        OCL_KERNEL kernel;
        bool useBias;
        int channel_in;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
        size_t weightSize;
        size_t biasSize;
    public:
        pwconv2d(layer* p, int u, bool use_bias = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_dwconv2d.cpp */
    class dwconv2d : public layer
    {
    protected:
        OCL_MEM wMEM = nullptr;
        OCL_MEM bMEM = nullptr;
        OCL_MEM iMEM = nullptr;
        OCL_KERNEL kernel;
        bool useBias;
        int padSize;
        int stride;
        int kSize;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
        size_t weightSize;
        size_t biasSize;
    public:
        dwconv2d(layer* p,int ks, int s = 1, int pad = 0, bool use_bias = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_dense.cpp */
    class dense : public layer
    {
    protected:
        OCL_MEM wMEM = nullptr;
        OCL_MEM bMEM = nullptr;
        OCL_MEM iMEM = nullptr;
        OCL_KERNEL kernel;
        bool useBias;
        int channel_in;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
        size_t weightSize;
        size_t biasSize;
    public:
        dense(layer* p, size_t u, bool use_bias = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_relu.cpp */
    class relu : public layer
    {
    protected:
        OCL_KERNEL kernel;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
    public:
        relu(layer* p, bool inplace_ = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_relu6.cpp */
    class relu6 : public layer
    {
    protected:
        OCL_KERNEL kernel;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
    public:
        relu6(layer* p, bool inplace_ = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_softmax.cpp */
    class softmax : public layer
    {
    protected:
        OCL_KERNEL kernel;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
    public:
        softmax(layer* p, bool inplace_ = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_batchNorm.cpp */
    class batchNorm : public layer
    {
    protected:
        OCL_KERNEL kernel;
        OCL_MEM pMEM = nullptr;
        OCL_MEM iMEM = nullptr;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
        size_t paramSize = 0;
        float epsilon = 0;
    public:
        batchNorm(layer* p, float epsilon_ = 0.00001, bool inplace_ = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_add.cpp */
    class add : public layer
    {
    protected:
        OCL_KERNEL kernel;
        layer* scLayer;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
    public:
        add(layer* p, layer* sc, bool inplace_ = true);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();
    };


    /* layer_globalAvg.cpp */
    class globalAvg : public layer
    {
    protected:
        OCL_MEM iMEM;
        OCL_KERNEL kernel;
        size_t gws[3] = {0};
        size_t lws[3] = {0};
    public:
        globalAvg(layer* p);
        void run();
        size_t setParam(OCL_TYPE* buffer);
        void setKernel();

    };


    /* sequential.cpp */
    class sequential
    {
    protected:
        std::vector<layer*> seq;

    public:
        squentail(){ }
        void add(layer *l);
        void summary();
        size_t size_() { return seq.size(); }
        layer* operator[](int i);
        layer* back() { return seq.back(); } ;
        bool loadWeight(std::string f);
        void inference(OCL_TYPE* x, OCL_TYPE* y);
    };
}




#endif /* INCLUDE_OCL_H_ */
