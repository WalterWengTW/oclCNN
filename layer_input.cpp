#include "ocl.h"

ocl::input::input(shape_t s)
{
    name = "Input";
    prvLayer = nullptr;
    shape = s;
    s_ = s.N * s.C * s.H * s.W * sizeof(OCL_TYPE);
    gMEM = ocl::createBuffer(dContext, CL_MEM_READ_WRITE, s_);
}

void ocl::input::run()
{
    if(cMEM != nullptr)
    {
        ocl::writeBuffer(dQueue, gMEM, s_, cMEM);
        clFinish(dQueue);
    }


}

size_t ocl::input::setParam(OCL_TYPE* buffer) { return 0; };


void ocl::input::setKernel(){ return; }
