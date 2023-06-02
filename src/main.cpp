#include <iostream>
#include <iomanip>
#include <thread>
#include <future>
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"

int getImage(float* inputData, Dataset* data);
float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset* data, NeuralNetwork* net);

int main(int argc, char* argv[]) {
    unsigned int amountThreads = std::thread::hardware_concurrency() - 1;
    Dataset* data = new Dataset[amountThreads];
    if (argc == 2) {
        if (!atoi(argv[1])) {
            for (int i = 0; i < amountThreads; i++) {
                data[i].setData("dataset/training/train-labels.idx1-ubyte", "dataset/training/train-images.idx3-ubyte");
            }
        } else {
            for (int i = 0; i < amountThreads; i++) {
                data[i].setData("dataset/test/t10k-labels.idx1-ubyte", "dataset/test/t10k-images.idx3-ubyte");
            }
        }
    } else {
        std::cout << "Usage: .\\build.exe [0|1] (0: training; 1: test)" << std::endl;
        exit(69420);
    }

    unsigned int inputSize = 784;
    unsigned int outputSize = 10;
    std::vector<unsigned int> sizes;
    sizes.push_back(inputSize);
    sizes.push_back(16);
    sizes.push_back(16);
    sizes.push_back(outputSize);

    std::vector<NeuralNetwork*> nets;
    for (int i = 0; i < amountThreads; i++) {
        nets.push_back(new NeuralNetwork(sizes));
        nets[i]->setDataset(&data[i]);
        nets[i]->setName("mnist");
        if (!i) {
            nets[i]->initializeReLU();
            nets[i]->saveNetworkState();
        } else {
            nets[i]->loadNetworkState();
        }
    }
    
    int label = 0;
    float* inputData = new float[inputSize];

    int width = data[0].getWidth();
    int height = data[0].getHeight();
    int dataSize = data[0].getSize();

    unsigned int gradientSize = nets[0]->getGradientVecSize();
    float* gradientVec = new float[gradientSize]();
    float* correctData = new float[outputSize]();
    float error = 0;

    unsigned int partitionSize = dataSize / amountThreads;
    float** gradientList = new float*[amountThreads];
    std::future<float*>* future = new std::future<float*>[amountThreads];

    for (int epoch = 0; epoch < 1; epoch++) {
        for (int i = 0; i < amountThreads; i++) {
            future[i] = std::async(trainPartition, i, partitionSize, &data[i], nets[i]);
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
        nets[0]->updateWeightsAndBiases(0.05, gradientVec);
        nets[0]->saveNetworkState();

        for (int i = 0; i < amountThreads; i++) {
            nets[i]->loadNetworkState();
            delete[] gradientList[i];
        }
    }
    nets[0]->saveNetworkState();


    data[0].getImages()->seekg(16);
    data[0].getLabel()->seekg(8);

    label = getImage(inputData, &data[0]);
    nets[0]->propagate(inputData);

    float* lastLayer = nets[0]->getResults();
    for (int i = 0; i < outputSize; i++) {
        if (label != i) {
            std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "%" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "% <-" << std::endl;
        }
    }

    for (int i = 0; i < amountThreads; i++) {
        delete nets[i];
    }
    delete[] lastLayer;
    delete[] data;
    delete[] inputData;
    delete[] future;
    delete gradientList;
    
    return 0;
}

int getImage(float* inputData, Dataset* data) {
    unsigned int width = data->getWidth();
    unsigned int height = data->getHeight();

    unsigned char tmp;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data->getImages()->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
            inputData[i * width + j] = tmp / 255;
        }
    }
    data->getLabel()->read(reinterpret_cast<char*>(&inputData), sizeof(unsigned char));
    return tmp;
}

float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset* data, NeuralNetwork* net) {
    unsigned int startPos = partition * partitionSize;
    data->getLabel()->seekg(8 + startPos);
    data->getImages()->seekg(16 + (startPos * data->getWidth() * data->getHeight()));

    unsigned int gradientVecSize = net->getGradientVecSize();
    float* gradientVec = new float[gradientVecSize]();
    float* fInputData = new float[data->getWidth() * data->getHeight()];
    float* correctData = new float[gradientVecSize]();
    
    int dataSize = startPos + partitionSize;
    if (startPos + (2 * partitionSize) > data->getSize()) {
        dataSize = data->getSize();
    }
    
    int amountRun = 0;
    for (int i = startPos; i < dataSize; i++, amountRun++) {        
        int label = getImage(fInputData, data);

        correctData[label]++;
        net->propagate(fInputData);
        net->backPropagate(correctData, gradientVec);
        correctData[label]--;
    }

    for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] /= amountRun; }

    delete[] correctData;
    delete[] fInputData;

    return gradientVec;
}
