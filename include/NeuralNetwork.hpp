#pragma once

#include <vector>
#include <fstream>
#include "Layer.hpp"
#include "Dataset.hpp"

class NeuralNetwork {
private:
    Dataset* dataset;
    std::vector<Layer*> layers;
    std::fstream* netfile;
public:
    NeuralNetwork(const char* path);
    NeuralNetwork(unsigned char layerAmount, unsigned int* sizes);
    ~NeuralNetwork();

    float* getResults() const;
    unsigned int getGradientVecSize() const;
    void setDataset(Dataset* dataset);

    float ReLU(float val);
    float dReLU(float val);
    void softmax(unsigned char layer);

    void propagate(float* inputData);
    void backPropagate(float* correctData, float* gradientVec);
    void initializeReLU();
    void updateWeightsAndBiases(float* gradientVec);
    void saveNetworkState(const char* path);
    void loadNetworkState(const char* path);
};
