#include "ocl.h"

ocl::batchNorm::batchNorm(layer* p, float epsilon_, bool inplace_)
{
    name = "BatchNorm";
    prvLayer = p;
    epsilon = epsilon_;
    shape = prvLayer->getShape();

    if(inplace_)
        gMEM = prvLayer->getGPUMem();
    else
    {
        size_t memSize = shape.N * shape.C * shape.H * shape.W;
        gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    }
    int info[] = {shape.C, int(epsilon*1e7)};
    iMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*2, &info);
    paramSize = shape.C * 4;
    setNParam(paramSize);
    gws[0] = shape.W;
    gws[1] = shape.H;
    gws[2] = shape.N;
    ocl::getLocalSet(gws, lws);
}

void ocl::batchNorm::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 3, gws, 0);
    clFinish(dQueue);
}

size_t ocl::batchNorm::setParam(OCL_TYPE* buffer)
{
    pMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, paramSize * sizeof(OCL_TYPE), buffer);
    return paramSize;
}

void ocl::batchNorm::setKernel()
{
    kernel = ocl::createKernel(dProgram, "BatchNorm");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&iMEM, &prvMEM, &pMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM) ,sizeof(OCL_MEM)});

}


