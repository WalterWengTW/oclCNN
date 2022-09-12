#include "ocl.h"

ocl::dense::dense(layer* p, size_t u, bool use_bias)
{
    name = "Dense";
    useBias = use_bias;
    prvLayer = p;
    shape_t prvShape = prvLayer->getShape();
    channel_in = prvShape.H * prvShape.W * prvShape.C;
    shape.N = prvShape.N;
    shape.C = u;
    shape.H = 1;
    shape.W = 1;
    size_t memSize = shape.N * shape.C * shape.H * shape.W;
    gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    iMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &channel_in);
    weightSize = channel_in * shape.C;
    biasSize = shape.C;
    setNParam(weightSize + int(useBias) * biasSize);
    gws[0] = shape.C;
    gws[1] = shape.N;
    gws[2] = 0;
    ocl::getLocalSet(gws, lws);
}


void ocl::dense::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 2, gws, 0);
    clFinish(dQueue);
}


size_t ocl::dense::setParam(OCL_TYPE* buffer)
{
    wMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightSize * sizeof(OCL_TYPE), buffer);
    if(useBias)
        bMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, biasSize * sizeof(OCL_TYPE), buffer + weightSize);
    return weightSize + int(useBias) * biasSize;
}


void ocl::dense::setKernel()
{
    kernel = ocl::createKernel(dProgram, "Dense");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&iMEM, &prvMEM, &wMEM, &bMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM)});
}
