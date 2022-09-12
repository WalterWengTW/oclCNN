# OpenCL based CNN Inference Framework
##### This library is wrapped in the OpenCL API as a convolutional neural network inference framework. Based on Layer objects, various computing layers are established. Each layer can build the required memory space from the output shape of the previous layer.The weights of each layer are stored in a binary file with consecutively arranged values. When loading, take out the values according to the weights of each layer and put them into the memory.

