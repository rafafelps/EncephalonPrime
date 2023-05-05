#include "Layer.hpp"

Layer::Layer(unsigned int neuronAmount, Layer* prevLayer) :
size(neuronAmount) {
    neuron = new Neuron[this->getSize()];

    if (!prevLayer) { weigth = nullptr; bias = nullptr; }
    else {
        weigth = new float*[prevLayer->getSize()];
        int weightRows = prevLayer->getSize();
        for (int i = 0; i < weightRows; i++) {
            weigth[i] = new float[this->getSize()];
        }
        bias = new float[this->getSize()];
    }
}

Layer::~Layer() {
    delete neuron;
    
    if (weigth) {
        int weightRows = this->getSize();
        for (int i = 0; i < weightRows; i++) {
            delete weigth[i];
        }
    }
    delete weigth;

    delete bias;
}

unsigned int Layer::getSize() const {
    return this->size;
}

Neuron* Layer::getNeuron(unsigned int index) const {
    return &this->neuron[index];
}

float Layer::getWeight(unsigned int prevNeuron, unsigned int currNeuron) const {
    return this->weigth[prevNeuron][currNeuron];
}

float Layer::getBias(unsigned int index) const {
    return this->bias[index];
}

void Layer::setWeight(float val, unsigned int prevNeuron, unsigned int currNeuron) {
    this->weigth[prevNeuron][currNeuron] = val;
}

void Layer::setBias(float val, unsigned int index) {
    this->bias[index] = val;
}
