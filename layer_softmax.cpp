#include "ocl.h"

ocl::softmax::softmax(layer* p, bool inplace_)
{
    name = "Softmax";
    prvLayer = p;
    shape = prvLayer->getShape();
    if(inplace_)
        gMEM = prvLayer->getGPUMem();
    else
    {
        size_t memSize = shape.N * shape.C * shape.H * shape.W;
        gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    }
    gws[0] = shape.C*shape.H*shape.W;
    gws[1] = shape.N;
    gws[2] = 0;
    ocl::getLocalSet(gws, lws);
}


void ocl::softmax::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 1, gws, 0);
    clFinish(dQueue);
}


size_t ocl::softmax::setParam(OCL_TYPE* buffer) { return 0; }


void ocl::softmax ::setKernel()
{
    kernel = ocl::createKernel(dProgram, "Softmax");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&prvMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM)});
}

