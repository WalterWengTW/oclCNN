#include "ocl.h"

ocl::relu6::relu6(layer* p, bool inplace_)
{
    name = "ReLU6";
    prvLayer = p;
    shape = prvLayer->getShape();
    if(inplace_)
        gMEM = prvLayer->getGPUMem();
    else
    {
        size_t memSize = shape.N * shape.C * shape.H * shape.W;
        gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    }
    gws[0] = shape.N*shape.C*shape.H*shape.W;
    gws[1] = 0;
    gws[2] = 0;
    ocl::getLocalSet(gws, lws);
}


void ocl::relu6::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 1, gws, 0);
    clFinish(dQueue);
}


size_t ocl::relu6::setParam(OCL_TYPE* buffer) { return 0; }


void ocl::relu6::setKernel()
{
    kernel = ocl::createKernel(dProgram, "ReLU6");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&prvMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM)});
}

