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

    float* getResults() const;
    std::vector<Layer*> getLayers() const;
    void setDataset(Dataset* dataset);

    float ReLU(float val);
    float dReLU(float val);
    void softmax(float* results);

    void propagate(float* inputData);
    void backPropagate(float* correctData, std::vector<float*>* gradientList);
    void randomizeWeightsAndBiases();
    void updateWeightsAndBiases(float* negativeGradientVec);
};
