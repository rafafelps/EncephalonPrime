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
    mnist.setName("mnist");
    mnist.initializeReLU();
    
    unsigned char inputData = 0;
    float* fInputData = new float[784]();
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();

    unsigned int gradientVecSize = mnist.getGradientVecSize();

    for (int epoch = 0; epoch < 100; epoch++) {
        data.getImages()->seekg(16);
        data.getLabel()->seekg(8);

        for (int times = 0; times < 60; times++) {
            float* gradientVec = new float[gradientVecSize]();

            for (int image = 0; image < 1000; image++) {
                float* correctData = new float[outputSize]();

                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        data.getImages()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
                        fInputData[i * width + j] = inputData / 255;
                    }
                }

                data.getLabel()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
                correctData[inputData++];

                mnist.propagate(fInputData);
                mnist.backPropagate(correctData, gradientVec);

                delete[] correctData;
            }
            
            for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] /= 100000; }
            mnist.updateWeightsAndBiases(gradientVec);

            delete[] gradientVec;
        }
    }
    mnist.saveNetworkState();

    data.getImages()->seekg(16);
    data.getLabel()->seekg(8);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
            fInputData[i * width + j] = inputData / 255;
        }
    }

    data.getLabel()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
    mnist.propagate(fInputData);

    /*

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
            fInputData[i * width + j] = inputData / 255;
        }
    }*/

    float* lastLayer = mnist.getResults();
    for (int i = 0; i < outputSize; i++) {
        if (inputData != i) {
            std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "%" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "% <-" << std::endl;
        }
    }

    delete[] lastLayer;
    
    return 0;
}
