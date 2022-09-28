#include <iostream>
#include "ocl.h"
#include <time.h>
#include <opencv2/opencv.hpp>

int main()
{
    auto platforms = ocl::getPlatformID();
    OCL_PLATFORM dPlatform = platforms[0];

    auto devices = ocl::getDeviceID(dPlatform);
    dDevice = devices[0];


    dContext = ocl::createContext(dPlatform, dDevice);
    dQueue = ocl::createQueue(dContext, dDevice);

    dProgram = ocl::createProgramWithSrc(dContext, {"Program.cl"});
    ocl::buildProgram(dProgram, dDevice, "");

    ocl::sequential model;
    model.add(new ocl::input({1, 3, 224, 224}));
    model.add(new ocl::conv2d(model.back(), 32, 3, 2, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));

    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 16, false));
    model.add(new ocl::batchNorm(model.back()));


    model.add(new ocl::pwconv2d(model.back(), 96, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 2, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 24, false));
    model.add(new ocl::batchNorm(model.back()));

    model.add(new ocl::pwconv2d(model.back(), 144, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 24, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[16]));




    model.add(new ocl::pwconv2d(model.back(), 144, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 2, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 32, false));
    model.add(new ocl::batchNorm(model.back()));


    model.add(new ocl::pwconv2d(model.back(), 192, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 32, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[33]));


    model.add(new ocl::pwconv2d(model.back(), 192, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 32, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[42]));




    model.add(new ocl::pwconv2d(model.back(), 192, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 2, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 64, false));
    model.add(new ocl::batchNorm(model.back()));

    model.add(new ocl::pwconv2d(model.back(), 384, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 64, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[59]));

    model.add(new ocl::pwconv2d(model.back(), 384, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 64, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[68]));

    model.add(new ocl::pwconv2d(model.back(), 384, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 64, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[77]));





    model.add(new ocl::pwconv2d(model.back(), 384, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 96, false));
    model.add(new ocl::batchNorm(model.back()));

    model.add(new ocl::pwconv2d(model.back(), 576, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 96, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[94]));

    model.add(new ocl::pwconv2d(model.back(), 576, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 96, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[103]));





    model.add(new ocl::pwconv2d(model.back(), 576, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 2, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 160, false));
    model.add(new ocl::batchNorm(model.back()));

    model.add(new ocl::pwconv2d(model.back(), 960, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 160, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[120]));

    model.add(new ocl::pwconv2d(model.back(), 960, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 160, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::add(model.back(), model[129]));




    model.add(new ocl::pwconv2d(model.back(), 960, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::dwconv2d(model.back(), 3, 1, 1, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));
    model.add(new ocl::pwconv2d(model.back(), 320, false));
    model.add(new ocl::batchNorm(model.back()));


    model.add(new ocl::pwconv2d(model.back(), 1280, false));
    model.add(new ocl::batchNorm(model.back()));
    model.add(new ocl::relu6(model.back()));


    model.add(new ocl::globalAvg(model.back()));

    model.add(new ocl::dense(model.back(), 2));
    //model.add(new ocl::softmax(model.back()));

    model.summary();

    model.loadWeight("detector.bin");


    cv::Mat image = cv::imread("./test/0562_1_2.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat If32;
    image.convertTo(If32, CV_32FC3);

    OCL_TYPE* predict = new OCL_TYPE [2];
    OCL_TYPE* data = new OCL_TYPE[3*224*224];

    for(int c = 0; c < image.channels(); ++c)
    {
        for(int y = 0; y < image.rows; ++y)
        {
            for(int x = 0; x < image.cols; ++x)
                data[c*image.rows*image.cols+y*image.cols+x] = (OCL_TYPE)If32.at<cv::Vec3f>(y, x)[c]/127.5 - 1;
        }
    }


    model.inference(data, predict);

    for(int i = 0; i < 2; ++i)
        printf("[%d] : %02.8f\n",i,predict[i]);











    return 0;
}
