#pragma once
#include "Neuron.hpp"

class Layer {
private:
    Neuron* neuron;
    float** weigth;
    float* bias;
    const unsigned int size;
public:
    Layer(unsigned int neuronAmount, Layer* prevLayer);
    ~Layer();

    unsigned int getSize() const;
    Neuron* getNeuron(unsigned int index) const;
    float getWeight(unsigned int prevNeuron, unsigned int currNeuron) const;
    float getBias(unsigned int index) const;
};
