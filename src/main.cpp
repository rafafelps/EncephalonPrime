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

    unsigned int inputSize = 2;
    unsigned int outputSize = 2;
    unsigned int sizes[3] = {inputSize, 3, outputSize};
    NeuralNetwork mnist(3, sizes);
    mnist.setDataset(&data);
    mnist.initializeReLU();
    
    unsigned char inputData = 0;
    float fInputData[784] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();

    unsigned int gradientVecSize = mnist.getGradientVecSize();

    std::vector<Layer*> layers = mnist.getLayers();
    layers[1]->setWeight(-0.17628916, 0, 0);
    layers[1]->setWeight(0.60783723, 1, 0);
    layers[1]->setWeight(-0.82100356, 0, 1);
    layers[1]->setWeight(-0.07367439, 1, 1);
    layers[1]->setWeight(1.73245199, 0, 2);
    layers[1]->setWeight(-0.44930451, 1, 2);

    layers[2]->setWeight(0.15498288, 0, 0);
    layers[2]->setWeight(0.53511806, 1, 0);
    layers[2]->setWeight(-0.19673162, 2, 0);
    layers[2]->setWeight(0.67081392, 0, 1);
    layers[2]->setWeight(-0.91220382, 1, 1);
    layers[2]->setWeight(-0.52278519, 2, 1);
    float* correctData = new float[10]();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data.getImages()->read((char*)(&inputData), sizeof(unsigned char));
            fInputData[i * width + j] = inputData / 255;
        }
    }
    data.getLabel()->read((char*)(&inputData), sizeof(unsigned char));
    correctData[inputData]++;
    
    for (int epoch = 0; epoch < 100; epoch++) {
        float* gradientVec = new float[gradientVecSize]();
        mnist.propagate(fInputData);
        mnist.backPropagate(correctData, gradientVec);
        for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] *= 0.0001; }
        mnist.updateWeightsAndBiases(gradientVec);
        delete[] gradientVec;
    }
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
