#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"
#include <iostream>

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

    unsigned int sizes[4] = {784, 16, 16, 10};
    NeuralNetwork mnist(4, sizes);
    mnist.setDataset(&data);
    
    unsigned char inputData[784] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read((char *)(&inputData[i * width + j]), sizeof(unsigned char));
        }
    }

    float fInputData[784] = {0};
    for (int i = 0; i < 784; i++) {
        fInputData[i] = inputData[i] / 255;
    }
    
    mnist.randomizeWeightsAndBiases();
    mnist.propagate(fInputData);
    float* lastLayer = mnist.getResults();

    for (int i = 0; i < 10; i++) {
        std::cout << lastLayer[i] << std::endl;
    }
    
    return 0;
}
