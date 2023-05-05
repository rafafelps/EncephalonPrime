#include <cstdlib>
#include <ctime>
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const char* path) {

}

NeuralNetwork::NeuralNetwork(unsigned char layerAmount, unsigned int* sizes) {
    this->layers.push_back(new Layer(sizes[0], NULL));
    for (int i = 1; i < layerAmount; i++) {
        this->layers.push_back(new Layer(sizes[i], layers[i-1]));
    }
}

NeuralNetwork::~NeuralNetwork() {
    unsigned char size = layers.size();
    for (int i = 0; i < size; i++) {
        delete layers[i];
    }
    layers.clear();
}

float* NeuralNetwork::getResults() const {
    unsigned char amountLayers = this->layers.size();
    unsigned int layerSize = this->layers[amountLayers - 1]->getSize();

    float* endVec = new float[layerSize];
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        endVec[currNeuron] = this->layers[amountLayers - 1]->getNeuron(currNeuron)->getValue();
    }

    return endVec;
}

void NeuralNetwork::setDataset(Dataset* dataset) {
    this->dataset = dataset;
}

float NeuralNetwork::ReLU(float val) {
    return (val > 0) ? val : 0;
}

float NeuralNetwork::dReLU(float val) {
    return (val > 0) ? 1 : 0;
}

void NeuralNetwork::propagate(float* inputData) {
    unsigned char currLayer = 0;
    unsigned char amountLayers = layers.size() - 1;
    unsigned int layerSize = this->layers[currLayer]->getSize();

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        this->layers[currLayer]->getNeuron(currNeuron)->setValue(inputData[currNeuron]);
    }
    currLayer++;

    while (currLayer < amountLayers) {
        layerSize = this->layers[currLayer]->getSize();
        int prevLayerSize = this->layers[currLayer-1]->getSize();

        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            float activationValue = 0;

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                activationValue += this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue() * 
                this->layers[currLayer]->getWeight(prevNeuron, currNeuron);
            }

            activationValue += this->layers[currLayer]->getBias(currNeuron);
            this->layers[currLayer]->getNeuron(currNeuron)->setValue(this->ReLU(activationValue));
        }
        currLayer++;
    }

    layerSize = this->layers[currLayer]->getSize();
    int prevLayerSize = this->layers[currLayer-1]->getSize();
    float activationValue[layerSize] = {0};
    float total = 0;
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            activationValue[currNeuron] += this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue() * 
            this->layers[currLayer]->getWeight(prevNeuron, currNeuron);
        }
        activationValue[currNeuron] += this->layers[currLayer]->getBias(currNeuron);
        total += activationValue[currNeuron];
    }

    if (!total) { exit(10101010); }

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        this->layers[currLayer]->getNeuron(currNeuron)->setValue(activationValue[currNeuron] / total);
    }
}

void NeuralNetwork::backPropagate(float* correctData, std::vector<float*>* gradientList) {

}

void NeuralNetwork::randomizeWeightsAndBiases() {
    srand(time(NULL));

    unsigned char currLayer = 1;
    unsigned char amountLayer = this->layers.size();

    while (currLayer < amountLayer) {
        int layerSize = this->layers[currLayer]->getSize();
        int prevLayerSize = this->layers[currLayer-1]->getSize();

        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                this->layers[currLayer]->setWeight(r, prevNeuron, currNeuron);
            }
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            this->layers[currLayer]->setBias(r, currNeuron);
        }

        currLayer++;
    }
}

void NeuralNetwork::updateWeightsAndBiases(float* negativeGradientVec) {

}
