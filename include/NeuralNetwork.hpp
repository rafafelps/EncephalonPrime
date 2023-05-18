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

    std::vector<Layer*> getLayers() const;
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
};
