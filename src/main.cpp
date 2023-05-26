#include <iostream>
#include <iomanip>
#include <thread>
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"

int getImage(float* fInputData, Dataset* data);
float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset data);

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
    float* fInputData = new float[inputSize]();
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();

    unsigned int gradientVecSize = mnist.getGradientVecSize();
    
    unsigned int threadCount = std::thread::hardware_concurrency();
    //std::thread** threads = new std::thread*[threadCount];



    for (int epoch = 0; epoch < 1000; epoch++) {
        data.getImages()->seekg(16 + (width * height));
        data.getLabel()->seekg(9);

        /*for (int thr = 0; thr < threadCount; thr++) {
            
        }*/

        for (int image = 0; image < 1; image++) {
            float* correctData = new float[outputSize]();
            float* gradientVec = new float[gradientVecSize]();

            correctData[getImage(fInputData, &data)]++;

            mnist.propagate(fInputData);
            mnist.backPropagate(correctData, gradientVec);
            mnist.updateWeightsAndBiases(0.001, gradientVec);

            delete[] correctData;
            delete[] gradientVec;
        }
    }
    mnist.saveNetworkState();

    data.getImages()->seekg(16 + (width * height));
    data.getLabel()->seekg(9);

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

int getImage(float* fInputData, Dataset* data) {
    unsigned int width = data->getWidth();
    unsigned int height = data->getHeight();

    unsigned char inputData;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data->getImages()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
            fInputData[i * width + j] = inputData / 255;
        }
    }

    data->getLabel()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
    return inputData;
}

float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset data) {
    unsigned int startPos = partition * partitionSize;
    data.getLabel()->seekg(8 + startPos);
    data.getImages()->seekg(16 + (startPos * data.getWidth() * data.getHeight()));

}
