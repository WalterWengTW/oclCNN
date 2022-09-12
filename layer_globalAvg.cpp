#include "ocl.h"

ocl::globalAvg::globalAvg(layer* p)
{
    name = "Global Average";
    prvLayer = p;
    shape_t prvShape = prvLayer->getShape();
    shape.N = prvShape.N;
    shape.C = prvShape.C;
    shape.H = 1;
    shape.W = 1;
    int info[] = {prvShape.H, prvShape.W, prvShape.C};
    size_t memSize = shape.N * shape.C * shape.H * shape.W;
    gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, sizeof(OCL_TYPE) * memSize);
    iMEM = ocl::createBuffer(dContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*3, info);
    gws[0] = shape.C;
    gws[1] = shape.N;
    ocl::getLocalSet(gws, lws);
}


void ocl::globalAvg::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 2, gws, 0);
    clFinish(dQueue);
}


size_t ocl::globalAvg::setParam(OCL_TYPE* buffer) { return 0; }


void ocl::globalAvg::setKernel()
{
    kernel = ocl::createKernel(dProgram, "GlobalAvg");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    ocl::setArg(kernel,
                {&iMEM, &prvMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM)});

}
