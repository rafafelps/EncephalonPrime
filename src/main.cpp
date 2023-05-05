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

    unsigned int sizes[3] = {3, 4, 2};
    NeuralNetwork mnist(3, sizes);
    mnist.setDataset(&data);
    
    unsigned char inputData[4] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();

    float fInputData[3] = {0};
    for (int i = 0; i < 3; i++) {
        fInputData[i] = 0;
    }
    
    mnist.randomizeWeightsAndBiases();
    mnist.propagate(fInputData);
    float* lastLayer = mnist.getResults();

    std::vector<Layer*> layers = mnist.getLayers();

    std::cout << layers[1]->getNeuron(0)->getValue() * layers[2]->getWeight(0,0) + layers[1]->getNeuron(1)->getValue() * layers[2]->getWeight(1,0) + layers[2]->getBias(0) << std::endl;
    std::cout << layers[1]->getNeuron(0)->getValue() * layers[2]->getWeight(0,1) + layers[1]->getNeuron(1)->getValue() * layers[2]->getWeight(1,1) + layers[2]->getBias(1) << std::endl;
    std::cout << lastLayer[0] << std::endl;
    std::cout << lastLayer[1] << std::endl;
    
    return 0;
}
