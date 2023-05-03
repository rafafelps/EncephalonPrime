#pragma once

#include "NeuralNetwork.hpp"

class Trainer {
private:
    NeuralNetwork* neuralNetwork;
    float* gradientVector;
public:
    Trainer();
    ~Trainer();

    void setNeuralNetwork(NeuralNetwork* neuralNetwork);

    void propagate(unsigned char* inputData);
    void backPropagate();
    float ReLU(float val);
};
