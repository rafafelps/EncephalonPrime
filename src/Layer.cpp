#include "Layer.hpp"

Layer::Layer(unsigned int neuronAmount, Layer* prevLayer) :
size(neuronAmount) {
    neuron = new Neuron[this->getSize()];

    if (!prevLayer) { weight = nullptr; bias = nullptr; }
    else {
        weight = new float*[neuronAmount];
        int weightColumns = prevLayer->getSize();
        for (int i = 0; i < neuronAmount; i++) {
            weight[i] = new float[weightColumns];
        }
        bias = new float[this->getSize()];
    }
}

Layer::~Layer() {
    delete[] neuron;
    
    if (weight) {
        int weightRows = this->getSize();
        for (int i = 0; i < weightRows; i++) {
            delete[] weight[i];
        }
    }
    delete[] weight;

    delete[] bias;
}

unsigned int Layer::getSize() const {
    return this->size;
}

Neuron* Layer::getNeuron(unsigned int index) const {
    return &this->neuron[index];
}

float Layer::getWeight(unsigned int prevNeuron, unsigned int currNeuron) const {
    return this->weight[currNeuron][prevNeuron];
}

float Layer::getBias(unsigned int index) const {
    return this->bias[index];
}

void Layer::setWeight(float val, unsigned int prevNeuron, unsigned int currNeuron) {
    this->weight[currNeuron][prevNeuron] = val;
}

void Layer::setBias(float val, unsigned int index) {
    this->bias[index] = val;
}
