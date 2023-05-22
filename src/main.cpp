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
    mnist.initializeReLU();
    
    unsigned char inputData = 0;
    float fInputData[784] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();

    unsigned int gradientVecSize = mnist.getGradientVecSize();

    std::vector<Layer*> layers = mnist.getLayers();
    /*layers[1]->setWeight(1.24, 0, 0);
    layers[1]->setWeight(0.91, 1, 0);
    layers[1]->setWeight(-0.73, 0, 1);
    layers[1]->setWeight(0.32, 1, 1);
    layers[1]->setWeight(-0.45, 0, 2);
    layers[1]->setWeight(-0.16, 1, 2);

    layers[2]->setWeight(-0.71, 0, 0);
    layers[2]->setWeight(0.64, 1, 0);
    layers[2]->setWeight(-0.87, 2, 0);
    layers[2]->setWeight(-0.18, 0, 1);
    layers[2]->setWeight(0.55, 1, 1);
    layers[2]->setWeight(0.19, 2, 1);*/
    float* correctData = new float[10]();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read((char*)(&inputData), sizeof(unsigned char));
            fInputData[i * width + j] = inputData / 255;
        }
    }
    float* gradientVec = new float[gradientVecSize]();
    data.getLabel()->read((char*)(&inputData), sizeof(unsigned char));
    correctData[inputData]++;
    mnist.propagate(fInputData);
    mnist.backPropagate(correctData, gradientVec);
    mnist.updateWeightsAndBiases(gradientVec);
    for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] = 0; }
    mnist.propagate(fInputData);
    mnist.backPropagate(correctData, gradientVec);
    mnist.updateWeightsAndBiases(gradientVec);
    mnist.propagate(fInputData);
    for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] = 0; }
    mnist.propagate(fInputData);
    mnist.backPropagate(correctData, gradientVec);
    mnist.updateWeightsAndBiases(gradientVec);
    mnist.propagate(fInputData);
    /*

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read((char*)(&inputData), sizeof(unsigned char));
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
