#pragma once
#include "Neuron.hpp"

class Layer {
private:
    Neuron* neuron;
    float** weight;
    float* bias;
    const unsigned int size;
public:
    Layer(unsigned int neuronAmount, Layer* prevLayer);
    ~Layer();

    unsigned int getSize() const;
    Neuron* getNeuron(unsigned int index) const;
    float getWeight(unsigned int prevNeuron, unsigned int currNeuron) const;
    float getBias(unsigned int index) const;

    void setWeight(float val, unsigned int prevNeuron, unsigned int currNeuron);
    void setBias(float val, unsigned int index);
};
