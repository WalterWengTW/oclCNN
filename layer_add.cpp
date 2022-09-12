#include "ocl.h"

ocl::add::add(layer* p, layer* sc, bool inplace_)
{
    name = "Add";
    prvLayer = p;
    scLayer = sc;
    shape = prvLayer->getShape();
    shape_t pShape = shape;
    shape_t scShape = scLayer->getShape();
    if(pShape.H != scShape.H || pShape.W != scShape.W || pShape.C != scShape.C || pShape.N != scShape.N)
    {
        printf("Short cut feature map size doesn't match.\n");
        assert(0);
    }
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


void ocl::add::run()
{
    ocl::ndRangeKernel(dQueue, kernel, 1, gws, 0);
    clFinish(dQueue);
}


size_t ocl::add::setParam(OCL_TYPE* buffer) { return 0; }


void ocl::add::setKernel()
{
    kernel = ocl::createKernel(dProgram, "Add");
    OCL_MEM prvMEM = prvLayer->getGPUMem();
    OCL_MEM scMEM  = scLayer->getGPUMem();
    ocl::setArg(kernel,
                {&prvMEM, &scMEM, &gMEM},
                {sizeof(OCL_MEM), sizeof(OCL_MEM), sizeof(OCL_MEM)});
}
