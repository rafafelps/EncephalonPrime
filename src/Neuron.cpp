#include "Neuron.hpp"

Neuron::Neuron(float value) {
    setValue(value);
}

Neuron::~Neuron() {}

float Neuron::getValue() const {
    return this->value;
}

void Neuron::setValue(float value) {
    this->value = value;
}
