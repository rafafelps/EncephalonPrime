#pragma once

#include <vector>
#include "Layer.hpp"
#include "Dataset.hpp"

class NeuralNetwork {
private:
    Dataset* dataset;
    std::vector<Layer*> layers;
public:
    NeuralNetwork(const char* path);
    NeuralNetwork(unsigned char layerAmount, unsigned int* sizes);
    ~NeuralNetwork();

    void setDataset(Dataset* dataset);
};
