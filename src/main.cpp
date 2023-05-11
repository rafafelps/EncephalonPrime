#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    Dataset data;
    if (argc == 2) {
        if (!atoi(argv[1])) {
            data.setData("dataset/training/train-labels.idx1-ubyte", "dataset/training/train-images.idx3-ubyte");
        } else {
            data.setData("dataset/test/t10k-labels.idx1-ubyte", "dataset/test/t10k-images.idx3-ubyte");
        }
    } else {
        std::cout << "Usage: .\\build.exe [0|1] (0: training; 1: test)" << std::endl;
        exit(69420);
    }

    unsigned int inputSize = 784;
    unsigned int outputSize = 10;
    unsigned int sizes[4] = {inputSize, 16, 16, outputSize};
    NeuralNetwork mnist(4, sizes);
    mnist.setDataset(&data);
    
    unsigned char inputData[784] = {0};
    float fInputData[784] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read((char *)(&inputData[i * width + j]), sizeof(unsigned char));
            fInputData[i * width + j] = inputData[i * width + j] / 255;
        }
    }
    
    mnist.initializeReLU();
    mnist.propagate(fInputData);
    float* lastLayer = mnist.getResults();

    std::vector<float*> gradientList;
    float correctData[10] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};

    //mnist.backPropagate(correctData, &gradientList);

    for (int i = 0; i < outputSize; i++) {
        std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "%" << std::endl;
    }
    
    return 0;
}
