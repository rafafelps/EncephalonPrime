#include <iostream>
#include <iomanip>
#include <thread>
#include <future>
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"

int getImage(float* fInputData, Dataset* data);
float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset data, NeuralNetwork net);

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
    unsigned int sizes[3] = {inputSize, 100, outputSize};
    NeuralNetwork mnist(3, sizes);
    mnist.setDataset(&data);
    mnist.setName("mnist");
    mnist.initializeReLU();
    
    int label = 0;
    float* fInputData = new float[inputSize];
    int width = data.getWidth();
    int height = data.getHeight();
    int dataSize = data.getSize();

    unsigned int gradientSize = mnist.getGradientVecSize();
    float* gradientVec = new float[gradientSize]();
    float* correctData = new float[outputSize]();
    float error = 0;

    unsigned int amountThreads = std::thread::hardware_concurrency() - 1;
    unsigned int partitionSize = dataSize / amountThreads;
    float** gradientList = new float*[amountThreads];
    std::future<float*>* future = new std::future<float*>[amountThreads];

    for (int epoch = 0; epoch < 5; epoch++) {
        for (int i = 0; i < amountThreads; i++) {
            future[i] = std::async(trainPartition, i, partitionSize, data, mnist);
        }

        for (int i = 0; i < amountThreads; i++) {
            gradientList[i] = future[i].get();
        }

        float* gradientVec = new float[gradientSize]();
        for (int i = 0; i < gradientSize; i++) {
            for (int j = 0; j < amountThreads; j++) {
                unsigned int deltaPartition;
                if (j == amountThreads - 1) {
                    deltaPartition = j * partitionSize;
                    deltaPartition = dataSize - 1 - deltaPartition;
                } else {
                    deltaPartition = partitionSize;
                }
                gradientVec[i] = (deltaPartition / dataSize) * gradientList[j][i];
            }
        }
        mnist.updateWeightsAndBiases(0.01, gradientVec);

        for (int i = 0; i < amountThreads; i++) { delete[] gradientList[i]; }
    }
    mnist.saveNetworkState();
    delete[] future;
    delete gradientList;

    data.getImages()->seekg(16);
    data.getLabel()->seekg(8);

    label = getImage(fInputData, &data);
    mnist.propagate(fInputData);

    float* lastLayer = mnist.getResults();
    for (int i = 0; i < outputSize; i++) {
        if (label != i) {
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

float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset data, NeuralNetwork net) {
    unsigned int startPos = partition * partitionSize;
    data.getLabel()->seekg(8 + startPos);
    data.getImages()->seekg(16 + ((startPos - 1) * data.getWidth() * data.getHeight()));

    unsigned int gradientVecSize = net.getGradientVecSize();
    float* gradientVec = new float[gradientVecSize]();
    float* fInputData = new float[data.getWidth() * data.getHeight()];
    float* correctData = new float[gradientVecSize]();
    
    int dataSize = startPos + partitionSize;
    if (startPos + (2 * partitionSize)) {
        dataSize = data.getSize();
    }
    
    int amountRun = 0;
    for (int i = startPos; i < dataSize; i++, amountRun++) {
        int label = getImage(fInputData, &data);

        correctData[label]++;
        net.propagate(fInputData);
        net.backPropagate(correctData, gradientVec);
        correctData[label]--;
    }

    for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] /= amountRun; }

    delete[] correctData;
    delete[] fInputData;

    return gradientVec;
}
