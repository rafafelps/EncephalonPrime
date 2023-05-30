#pragma once

#include <vector>
#include "Layer.hpp"
#include "Dataset.hpp"

class NeuralNetwork {
private:
    Dataset* dataset;
    std::vector<Layer*> layers;
    std::string name;
public:
    NeuralNetwork(std::string name);
    NeuralNetwork(unsigned int layerAmount, unsigned int* sizes);
    ~NeuralNetwork();

    float* getResults() const;
    unsigned int getGradientVecSize() const;
    float getError(unsigned int correctResult) const;
    void setDataset(Dataset* dataset);
    void setName(std::string name);

    float ReLU(float val);
    float dReLU(float val);
    void softmax(unsigned char layer);

    void propagate(float* inputData);
    void backPropagate(float* correctData, float* gradientVec);
    void initializeReLU();
    void updateWeightsAndBiases(float learningRate, float* gradientVec);
    void saveNetworkState();
    void loadNetworkState();
};
