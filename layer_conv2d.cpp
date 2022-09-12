#include "ocl.h"

ocl::conv2d::conv2d(layer* p, int u, int ks, int s, int pad, bool use_bias)
{
    name = "Conv2D";
    useBias = use_bias;
    padSize = pad;
    stride = s>2?2:s;
    kSize = ks;
    prvLayer = p;
    shape_t prvShape = prvLayer->getShape();
    channel_in = prvShape.C;
    shape.N = prvShape.N;
    shape.C = u;
    shape.H = (prvShape.H - kSize + 2 * padSize)/stride + 1;
    shape.W = (prvShape.W - kSize + 2 * padSize)/stride + 1;

    int info[] = {prvShape.H, prvShape.W, prvShape.C, shape.H, shape.W, shape.C, stride, kSize, padSize};
    size_t memSize = shape.N * shape.C * shape.H * shape.W;
    gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    iMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*9, info);

    weightSize = shape.C * channel_in * kSize * kSize;
    biasSize = shape.C;
    setNParam(weightSize + int(useBias) * biasSize);
    gws[0] = shape.W;
    gws[1] = shape.H;
    gws[2] = shape.N;
    ocl::getLocalSet(gws, lws);
}


void ocl::conv2d::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 3, gws, 0);
    clFinish(dQueue);
}


size_t ocl::conv2d::setParam(OCL_TYPE* buffer)
{
    wMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightSize * sizeof(OCL_TYPE), buffer);
    if(useBias)
        bMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, biasSize * sizeof(OCL_TYPE), buffer + weightSize);
    return weightSize + int(useBias) * biasSize;
}


void ocl::conv2d::setKernel()
{
    kernel = ocl::createKernel(dProgram, "Conv2D");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&iMEM, &prvMEM, &wMEM, &bMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM)});
}

