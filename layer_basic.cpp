#include "ocl.h"

void ocl::layer::detach()
{
    cl_int ret;
    size_t bSize;
    ret = clGetMemObjectInfo(gMEM, CL_MEM_SIZE, sizeof(size_t), &bSize, NULL);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return;
    }
    else
    {
        if(cMEM == nullptr)
            cMEM = new OCL_TYPE [bSize/sizeof(OCL_TYPE)];
        ocl::readBuffer(dQueue, gMEM, bSize, cMEM);
    }
}
