#include "ocl.h"

void ocl::sequential::add(layer *l){ seq.push_back(l); }

ocl::layer* ocl::sequential::operator[](int i) { return seq[i]; }

void ocl::sequential::summary()
{
    size_t nParamAll = 0;
    printf("\n\nModel Summary\n\n");
    printf("#\tLayer\t\tOutput Shape[N,C,H,W]\t\tParameters\n");
    for(int i = 0; i < seq.size(); ++i)
    {
        printf("--------------------------------------------------------------------\n");
        shape_t s = seq[i]->getShape();
        std::string n = seq[i]->getName();
        size_t nParam = seq[i]->getNParam();
        nParamAll += nParam;
        printf("[%d]\t%-10s\t[%3d, %3d, %3d, %3d]\t\t%i\n", i, n.c_str(), s.N, s.C, s.H, s.W, nParam);
    }
    printf("--------------------------------------------------------------------\n\n");
    printf("                                                        %d\n\n", nParamAll);
}

bool ocl::sequential::loadWeight(std::string f)
{
    FILE *pFile = fopen(f.c_str(), "rb");
    size_t nParam;
    if(pFile == NULL)
    {
        printf("%s not found.\n", f.c_str());
        return false;
    }
    fseek(pFile, 0, SEEK_END);
    nParam = ftell(pFile)/sizeof(OCL_TYPE);
    rewind(pFile);
    OCL_TYPE* buffer = new OCL_TYPE [nParam];
    int nRead = fread(buffer, sizeof(OCL_TYPE), nParam, pFile);
    fclose(pFile);
    if(nRead != nParam)
    {
        delete [] buffer;
        printf("%s : Number of byte is not correct.\n", f.c_str());
        return 0;
    }

    int accum = 0;
    for(int i = 0; i < seq.size(); ++i)
    {
        int loaded = seq[i]->setParam(buffer + accum);
        accum += loaded;
        if(accum > nParam)
        {
            printf("Weighting over load.\n");
            delete [] buffer;
            return false;
        }
        seq[i]->setKernel();
    }
    printf("Model deployed.\n");
    delete [] buffer;
    return true;
}


void ocl::sequential::inference(OCL_TYPE* x, OCL_TYPE* y)
{
    seq[0]->setCPUMem(x);
    seq.back()->setCPUMem(y);
    for(int i = 0; i < seq.size(); ++i)
        seq[i]->run();
    seq.back()->detach();
}
