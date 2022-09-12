#include "ocl.h"

OCL_DEVICE dDevice;
OCL_CONTEXT dContext;
OCL_QUEUE dQueue;
OCL_PROGRAM dProgram;


std::vector<OCL_PLATFORM> ocl::getPlatformID()
{
    cl_int ret;
    cl_uint nPlatform = 0;
    ret = clGetPlatformIDs(0, 0, &nPlatform);
    if(ret!=CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return {};
    }
    std::vector<OCL_PLATFORM> ids(nPlatform);
    ret = clGetPlatformIDs(nPlatform, &ids[0], NULL);
    if(ret!=CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return {};
    }
    return ids;
}

std::vector<OCL_DEVICE> ocl::getDeviceID(OCL_PLATFORM p, cl_device_type type)
{
    cl_int ret;
    cl_uint nDevice = 0;
    ret = clGetDeviceIDs(p, type, 0, NULL, &nDevice);
    if(ret!=CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return {};
    }
    std::vector<OCL_DEVICE> ids(nDevice);
    ret = clGetDeviceIDs(p, type, nDevice, &ids[0], NULL);
    if(ret!=CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return {};
    }
    return ids;
}


OCL_CONTEXT ocl::createContext(OCL_PLATFORM p, OCL_DEVICE d)
{
    cl_int ret;
    OCL_CONTEXT ctxt = NULL;
    cl_context_properties contextPty[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0};
    ctxt = clCreateContext(contextPty, 1, &d, NULL, NULL, &ret);
    if(ret!=CL_SUCCESS)
        ocl::getErrMsg(ret);
    return ctxt;
}


OCL_QUEUE ocl::createQueue(OCL_CONTEXT c, OCL_DEVICE d)
{
    cl_int ret;
    OCL_QUEUE cmdQueue = NULL;
    cl_command_queue_properties queuePty[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cmdQueue = clCreateCommandQueueWithProperties(c, d, queuePty, &ret);
    if(ret!=CL_SUCCESS)
        ocl::getErrMsg(ret);
    return cmdQueue;
}


OCL_PROGRAM ocl::createProgramWithSrc(OCL_CONTEXT c, std::vector<std::string> fileNames)
{
    cl_int ret;
    OCL_PROGRAM program = NULL;
    cl_uint nProgram = fileNames.size();
    char** programs = new char* [nProgram];
    size_t *lengths = new size_t [nProgram];


    for(cl_uint i = 0; i < nProgram; ++i)
    {
        FILE *pFile = fopen(fileNames[i].c_str(), "rb");
        size_t nChar;

        if(pFile == NULL)
        {
            printf("%s not found.\n", fileNames[i].c_str());
            return 0;
        }
        fseek(pFile, 0, SEEK_END);
        nChar = ftell(pFile);
        rewind(pFile);
        programs[i] = new char [nChar];
        int nRead = fread(programs[i], sizeof(char), nChar, pFile);
        fclose(pFile);
        if(nRead != nChar)
        {
            delete [] programs[i];
            printf("%s : Number of character is not correct.\n", fileNames[i].c_str());
            return 0;
        }
        lengths[i] = nRead;
        //lengths[i] = ocl::getFileData(fileNames[i], programs[i]);
    }
    program = clCreateProgramWithSource(c, nProgram, (const char**)programs, &lengths[0], &ret);
    if(ret != CL_SUCCESS)
        ocl::getErrMsg(ret);


    for(cl_uint i = 0; i < nProgram; ++i)
        delete [] programs[i];
    delete [] programs;
    delete [] lengths;
    return program;
}


void ocl::buildProgram(OCL_PROGRAM p, OCL_DEVICE d, const char* options)
{
    cl_int ret;
    ret = clBuildProgram(p, 1, &d, options, NULL, NULL);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        ocl::getBuildInfo(p, d);
        return;
    }
}


OCL_KERNEL ocl::createKernel(OCL_PROGRAM p, const char* k)
{
    cl_int ret;
    OCL_KERNEL kernel = clCreateKernel(p, k, &ret);
    if(ret != CL_SUCCESS)
        ocl::getErrMsg(ret);
    return kernel;
}


OCL_MEM ocl::createBuffer(OCL_CONTEXT c, OCL_MFLAGS f, size_t s, void *h)
{
    cl_int ret;
    OCL_MEM mem = NULL;
    mem = clCreateBuffer(c, f, s, h, &ret);
    if(ret != CL_SUCCESS)
        ocl::getErrMsg(ret);
    return mem;
}
