#include "ocl.h"

ocl::pwconv2d::pwconv2d(layer* p, int u, bool use_bias)
{
    name = "PW-Conv2D";
    useBias = use_bias;
    prvLayer = p;
    shape_t prvShape = prvLayer->getShape();
    channel_in = prvShape.C;
    shape.N = prvShape.N;
    shape.C = u;
    shape.H = prvShape.H;
    shape.W = prvShape.W;

    int info[] = {prvShape.H, prvShape.W, prvShape.C, shape.H, shape.W, shape.C};
    size_t memSize = shape.N * shape.C * shape.H * shape.W;
    gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    iMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*6, info);

    weightSize = shape.C * channel_in;
    biasSize = shape.C;
    setNParam(weightSize + int(useBias) * biasSize);
    gws[0] = shape.W;
    gws[1] = shape.H;
    gws[2] = shape.N;
    ocl::getLocalSet(gws, lws);
}


void ocl::pwconv2d::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 3, gws, 0);
    clFinish(dQueue);
}


size_t ocl::pwconv2d::setParam(OCL_TYPE* buffer)
{
    wMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightSize * sizeof(OCL_TYPE), buffer);
    if(useBias)
        bMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, biasSize * sizeof(OCL_TYPE), buffer + weightSize);
    return weightSize + int(useBias) * biasSize;
}


void ocl::pwconv2d::setKernel()
{
    kernel = ocl::createKernel(dProgram, "PWConv2D");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&iMEM, &prvMEM, &wMEM, &bMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM)});
}

