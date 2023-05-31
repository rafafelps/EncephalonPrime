#include "Neuron.hpp"

Neuron::Neuron(float value) {
    setValue(value);
}

Neuron::~Neuron() {}

float Neuron::getValue() const {
    return value;
}

float Neuron::getActValue() const {
    return actValue;
}

void Neuron::setValue(float value) {
    this->value = value;
}

void Neuron::setActValue(float value) {
    actValue = value;
}
