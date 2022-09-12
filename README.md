# OpenCL based CNN Inference Framework
##### This library is wrapped in the OpenCL API as a convolutional neural network inference framework. Based on Layer objects, various computing layers are established. Each layer can build the required memory space from the output shape of the previous layer.The weights of each layer are stored in a binary file with consecutively arranged values. When loading, take out the values according to the weights of each layer and put them into the memory.

##### There is an example showing how to use this API
```c++=0
#include "ocl.h"

int main() {
    auto platforms = ocl::getPlatformID();
    OCL_PLATFORM dPlatform = platforms[0];  //Get one platform id

    auto devices = ocl::getDeviceID(dPlatform);
    dDevice = devices[0];   //Get one device id to be only one device used.

    dContext = ocl::createContext(dPlatform, dDevice);   // Create context
    dQueue = ocl::createQueue(dContext, dDevice);        // Create command queue

    dProgram = ocl::createProgramWithSrc(dContext, {"Program.cl"});   //Build kernel program for GPU computing
    ocl::buildProgram(dProgram, dDevice, "");
    
    //Create a model object, and add the layers all you need. Layer are created by dynamically declared, and connect to previous layer.
    ocl::sequential model;
    model.add(new ocl::input({1, 1, 28, 28}));
    model.add(new ocl::conv2d(model.back(), 6, 5, 1, 2, false));
    model.add(new ocl::relu(model.back()));

    model.add(new ocl::conv2d(model.back(), 6, 5, 2, 2, false));
    model.add(new ocl::relu(model.back()));

    model.add(new ocl::conv2d(model.back(), 16, 5, 1, 0, false));
    model.add(new ocl::relu(model.back()));

    model.add(new ocl::conv2d(model.back(), 16, 5, 2, 2, false));
    model.add(new ocl::relu(model.back()));

    model.add(new ocl::conv2d(model.back(), 120, 5, 1, 0, false));
    model.add(new ocl::relu(model.back()));

    model.add(new ocl::dense(model.back(), 10));
    model.add(new ocl::softmax(model.back()));

    // Show information of each layer
    model.summary();

    // Load pre-trained model weightings from .bin file
    model.loadWeight("mnistCNN.bin");


    cv::Mat image = cv::imread("./test/9.png", cv::IMREAD_GRAYSCALE);
    cv::Mat Inorm;
    image.convertTo(Inorm, CV_32F);
    Inorm /= 255;

    OCL_TYPE* data = (OCL_TYPE*)Inorm.ptr();
    OCL_TYPE* predict = new OCL_TYPE [10];

    model.inference(data, predict);

    for(int i = 0; i < 10; ++i)
        printf("[%d] : %02.8f %%\n",i,predict[i]*100);
}

```
