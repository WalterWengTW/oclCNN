#include "ocl.h"

void ocl::getPlatformName(OCL_PLATFORM p)
{
    cl_int ret;
    ::size_t nInfo;
    ret = clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, NULL, &nInfo);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return;
    }
    char *sInfo = new char[nInfo];
    ret = clGetPlatformInfo(p, CL_PLATFORM_NAME, nInfo, sInfo, NULL);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return;
    }
    printf("Platform Name : %s\n", sInfo);
    delete [] sInfo;
}


void ocl::getDeviceName(OCL_DEVICE d)
{
    cl_int ret;
    ::size_t nInfo;
    ret = clGetDeviceInfo(d, CL_DEVICE_NAME, 0, NULL, &nInfo);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return;
    }
    char *sInfo = new char[nInfo];
    ret = clGetDeviceInfo(d, CL_DEVICE_NAME, nInfo, sInfo, NULL);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return;
    }
    printf("Device Name : %s\n",  sInfo);
    delete [] sInfo;
}


size_t ocl::getFileData(std::string &fileName, char* s)
{
    FILE *pFile = fopen(fileName.c_str(), "rb");
    size_t nChar;

    if(pFile == NULL)
    {
        printf("%s not found.\n", fileName.c_str());
        return 0;
    }
    fseek(pFile, 0, SEEK_END);
    nChar = ftell(pFile);
    rewind(pFile);
    s = new char [nChar];
    int nRead = fread(s, sizeof(char), nChar, pFile);
    fclose(pFile);
    if(nRead != nChar)
    {
        delete [] s;
        printf("%s : Number of character is not correct.\n", fileName.c_str());
        return 0;
    }

    return nRead;
}


void ocl::getBuildInfo(OCL_PROGRAM p, OCL_DEVICE d)
{
    cl_int ret;
    size_t nInfo;
    ret = clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, 0, NULL, &nInfo);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        return;
    }
    char *pInfo = new char [nInfo];
    ret = clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, nInfo, pInfo, NULL);
    if(ret != CL_SUCCESS)
    {
        ocl::getErrMsg(ret);
        delete [] pInfo;
    }
    else
    {
        printf("--------------- Compiler Log ---------------\n\n%s", pInfo);
        delete [] pInfo;
    }
}


void ocl::setArg(OCL_KERNEL k, std::vector<void*> args, std::vector<size_t> argSizes)
{
    cl_int ret;
    size_t nArgs = args.size();
    if(nArgs != argSizes.size())
    {
        printf("Number of argument is not match argument size.\n");
        return;
    }

    for(int i = 0; i < nArgs; ++i)
    {
        ret = clSetKernelArg(k, i, argSizes[i], args[i]);
        if(ret != CL_SUCCESS)
        {
            printf("Arg[%d] : ");
            ocl::getErrMsg(ret);
            return;
        }
    }
}


void ocl::writeBuffer(OCL_QUEUE q, OCL_MEM m, size_t s, void *ptr, OCL_EVENT *e)
{
    cl_int ret;
    ret = clEnqueueWriteBuffer(q, m, CL_TRUE, 0, s, ptr, 0, NULL, e);
    if(ret != CL_SUCCESS)
        ocl::getErrMsg(ret);
}


void ocl::readBuffer(OCL_QUEUE q, OCL_MEM m, size_t s, void *ptr, OCL_EVENT *e)
{
    cl_int ret;
    ret = clEnqueueReadBuffer(q, m, CL_TRUE, 0, s, ptr, 0, NULL, e);
    if(ret != CL_SUCCESS)
        ocl::getErrMsg(ret);
}


void ocl::ndRangeKernel(OCL_QUEUE q, OCL_KERNEL k, size_t d, const size_t* gws, const size_t* lws, OCL_EVENT *e)
{
    cl_int ret;
    ret = clEnqueueNDRangeKernel(q, k, d, NULL, gws, lws, 0, NULL, e);
    if(ret != CL_SUCCESS)
        ocl::getErrMsg(ret);

}

void ocl::getErrMsg(cl_int e)
{
    switch(e)
    {
        case CL_SUCCESS:
            printf("Success.\n");
            break;
        case CL_DEVICE_NOT_FOUND:
            printf("Device not found.\n");
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            printf("Device not available.\n");
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            printf("Compiler not available.\n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            printf("Memory object allocation failed.\n");
            break;
        case CL_OUT_OF_RESOURCES:
            printf("Out of resource.\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            printf("Out of host memory.\n");
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            printf("Profiling information not available.\n");
            break;
        case CL_MEM_COPY_OVERLAP:
            printf("Memory copy overlap.\n");
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            printf("Image format mismatch.\n");
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            printf("Image format not supported.\n");
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            printf("Build program failed.\n");
            break;
        case CL_MAP_FAILURE:
            printf("Mapping failed.\n");
            break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            printf("Misaligned sub-buffer offset.\n");
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            printf("Execution status error for events in wait list.\n");
            break;
        case CL_COMPILE_PROGRAM_FAILURE:
            printf("Compile program failed.\n");
            break;
        case CL_LINKER_NOT_AVAILABLE:
            printf("Linker not available.\n");
            break;
        case CL_LINK_PROGRAM_FAILURE:
            printf("Link program failed.\n");
            break;
        case CL_DEVICE_PARTITION_FAILED:
            printf("Device partition failed.\n");
            break;
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            printf("Kernel argument information not available.\n");
            break;
        case CL_INVALID_VALUE:
            printf("Invalid value.\n");
            break;
        case CL_INVALID_DEVICE_TYPE:
            printf("Invalid device type.\n");
            break;
        case CL_INVALID_PLATFORM:
            printf("Invalid platform.\n");
            break;
        case CL_INVALID_DEVICE:
            printf("Invalid device.\n");
            break;
        case CL_INVALID_CONTEXT:
            printf("Invalid context.\n");
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            printf("Invalid queue properties.\n");
            break;
        case CL_INVALID_COMMAND_QUEUE:
            printf("Invalid command queue.\n");
            break;
        case CL_INVALID_HOST_PTR:
            printf("Invalid host pointer.\n");
            break;
        case CL_INVALID_MEM_OBJECT:
            printf("Invalid memory object.\n");
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            printf("Invalid image format descriptor.\n");
            break;
        case CL_INVALID_IMAGE_SIZE:
            printf("Invalid image size.\n");
            break;
        case CL_INVALID_SAMPLER:
            printf("Invalid sampler.\n");
            break;
        case CL_INVALID_BINARY:
            printf("Invalid binary.\n");
            break;
        case CL_INVALID_BUILD_OPTIONS:
            printf("Invalid build options.\n");
            break;
        case CL_INVALID_PROGRAM:
            printf("Invalid program.\n");
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            printf("Invalid program executable.\n");
            break;
        case CL_INVALID_KERNEL_NAME:
            printf("Invalid kernel name.\n");
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            printf("Invalid kernel definition.\n");
            break;
        case CL_INVALID_KERNEL:
            printf("Invalid kernel.\n");
            break;
        case CL_INVALID_ARG_INDEX:
            printf("Invalid argument index.\n");
            break;
        case CL_INVALID_ARG_VALUE:
            printf("Invalid argument value.\n");
            break;
        case CL_INVALID_ARG_SIZE:
            printf("Invalid argument size.\n");
            break;
        case CL_INVALID_KERNEL_ARGS:
            printf("Invalid kernel arguments.\n");
            break;
        case CL_INVALID_WORK_DIMENSION:
            printf("Invalid work dimension.\n");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            printf("Invalid work group size.\n");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            printf("Invalid work time size.\n");
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            printf("Invalid global offset.\n");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            printf("Invalid event wait list.\n");
            break;
        case CL_INVALID_EVENT:
            printf("Invalid event.\n");
            break;
        case CL_INVALID_OPERATION:
            printf("Invalid operation.\n");
            break;
        case CL_INVALID_GL_OBJECT:
            printf("Invalid GL object.\n");
            break;
        case CL_INVALID_BUFFER_SIZE:
            printf("Invalid buffer size.\n");
            break;
        case CL_INVALID_MIP_LEVEL:
            printf("Invalid map level.\n");
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            printf("Invalid global work size.\n");
            break;
        case CL_INVALID_PROPERTY:
            printf("Invalid property.\n");
            break;
        case CL_INVALID_IMAGE_DESCRIPTOR:
            printf("Invalid image descriptor.\n");
            break;
        case CL_INVALID_COMPILER_OPTIONS:
            printf("Invalid compiler options.\n");
            break;
        case CL_INVALID_LINKER_OPTIONS:
            printf("Invalid linker options.\n");
            break;
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            printf("Invalid device partition count.\n");
            break;
        default:
            printf("Unknown error.\n");
            break;
    }
}


void ocl::getLocalSet(size_t gws[3], size_t lws[3])
{
    for(int i = gws[0]; i > 1; --i)
    {
        if(gws[0]%i == 0 && i <= 256)
        {
            if(gws[1]<=1)
            {
                lws[0] = i;
                i = 0;
            }
            else
            {
                for(int j = gws[1]; j > 1; --j)
                {
                    if(gws[1]%j == 0 && i*j<=256)
                    {
                        if(gws[2]<=1)
                        {
                            lws[0] = i;
                            lws[1] =j;
                            i = 0;
                            j = 0;
                        }
                        else
                        {
                            for(int k = gws[2]; k > 1; --k)
                            {
                                if(gws[2]%k == 0 && i*j*k<=256)
                                {
                                    lws[0] = i;
                                    lws[1] =j;
                                    lws[2] =k;
                                    i = 0;
                                    j = 0;
                                    k = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
