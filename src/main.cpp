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

    unsigned int batchSize = 1;
    unsigned int maxBatch = dataSize / batchSize;

    //mnist.loadNetworkState();
    for (int epoch = 0; epoch < 3; epoch++) {
        data.getImages()->seekg(16);
        data.getLabel()->seekg(8);
        for (int batch = 0; batch < maxBatch; batch++) {
            for (int i = 0; i < gradientSize; i++) { gradientVec[i] = 0; }

            for (int images = 0; images < batchSize; images++) {
                label = getImage(fInputData, &data);

                correctData[label]++;

                mnist.propagate(fInputData);
                error += mnist.getCost(label);
                mnist.backPropagate(correctData, gradientVec);

                correctData[label]--;
            }

            for (int i = 0; i < gradientSize; i++) { gradientVec[i] /= batchSize; }
            std::cout << "\rCost: " << error / batchSize << std::flush;
            error = 0;

            mnist.updateWeightsAndBiases(0.05, gradientVec);
        }
    }
    mnist.saveNetworkState();

    delete[] gradientVec;
    delete[] correctData;

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

float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset data) {
    unsigned int startPos = partition * partitionSize;
    data.getLabel()->seekg(8 + startPos);
    data.getImages()->seekg(16 + (startPos * data.getWidth() * data.getHeight()));

}
