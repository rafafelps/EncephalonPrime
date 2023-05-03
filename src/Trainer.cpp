#include "Trainer.hpp"

Trainer::Trainer() {}

Trainer::~Trainer() {}

void Trainer::setNeuralNetwork(NeuralNetwork* neuralNetwork) {
    this->neuralNetwork = neuralNetwork;
}
