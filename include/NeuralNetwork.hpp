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
    NeuralNetwork(std::vector<unsigned int> sizes);
    NeuralNetwork(NeuralNetwork* originalNet);
    NeuralNetwork();
    ~NeuralNetwork();

    float* getResults() const;
    unsigned int getGradientVecSize() const;
    float getCost(unsigned int correctResult) const;
    void setDataset(Dataset* dataset);
    void setName(std::string name);
    void setStructure(std::vector<unsigned int> sizes);

    void adam(unsigned int t, float* correctData, float* m, float* v, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    float ReLU(float val);
    float dReLU(float val);
    float sigmoid(float val);
    float dSigmoid(float val);
    void softmax(unsigned char layer);

    void learn(unsigned int epochs, bool loadFromFile);
    void propagate(float* inputData);
    void backPropagate(float* correctData, float* gradientVec);
    void kaimingInitialization();
    void updateWeightsAndBiases(float learningRate, float* gradientVec);
    void saveNetworkState();
    void loadNetworkState();
};
