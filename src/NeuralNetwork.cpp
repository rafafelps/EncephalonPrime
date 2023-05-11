#include <ctime>
#include <cmath>
#include <queue>
#include <random>
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

void NeuralNetwork::softmax(float* results) {
    unsigned char amountLayers = this->layers.size();
    unsigned int layerSize = this->layers[amountLayers-1]->getSize();

    float total = 0;
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        results[currNeuron] = expf(results[currNeuron]);
        total += results[currNeuron];
    }

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        results[currNeuron] = results[currNeuron] / total;
    }
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
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            activationValue[currNeuron] += this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue() * 
            this->layers[currLayer]->getWeight(prevNeuron, currNeuron);
        }
        activationValue[currNeuron] += this->layers[currLayer]->getBias(currNeuron);
    }

    softmax(activationValue);

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        this->layers[currLayer]->getNeuron(currNeuron)->setValue(activationValue[currNeuron]);
    }
}

void NeuralNetwork::backPropagate(float* correctData, std::vector<float*>* gradientList) {
    unsigned int gradientSize = 0;
    unsigned char amountLayers = this->layers.size();

    for (int currLayer = 1; currLayer < amountLayers; currLayer++) {
        gradientSize += this->layers[currLayer-1]->getSize() *
                        this->layers[currLayer]->getSize() +
                        this->layers[currLayer]->getSize();
    }

    float* gradientVec = new float[gradientSize];

    unsigned char currLayer = amountLayers - 1;
    unsigned int layerSize = this->layers[currLayer]->getSize();
    unsigned int prevLayerSize = this->layers[currLayer-1]->getSize();
    unsigned int gradientCounter = 0;
    std::queue<float> deltas;
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float dCdA = 2 *
                     (this->layers[currLayer]->getNeuron(currNeuron)->getValue()
                     - correctData[currNeuron]);
        deltas.push(dCdA);
        float dRelu = dReLU(this->layers[currLayer]->getNeuron(currNeuron)->getValue());
        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            gradientVec[gradientCounter++] = this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue() *
                                             dRelu * dCdA;
        }
        gradientVec[gradientCounter++] = dCdA * dRelu;
        // Add dCdA * dRelu to bias vec and add it later to gradientVec
    }

    currLayer--;
    while (currLayer > 0) {
        layerSize = this->layers[currLayer]->getSize();
        prevLayerSize = this->layers[currLayer-1]->getSize();
        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            float dCdA = 0;
            float dRelu = 0;
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                unsigned int nextLayerSize = this->layers[currLayer+1]->getSize();

                for (int nextNeuron = 0; nextNeuron < nextLayerSize; nextNeuron++) {
                    dCdA += this->layers[currLayer+1]->getWeight(currNeuron, nextNeuron) *
                            dReLU(this->layers[currLayer+1]->getNeuron(nextNeuron)->getValue()) *
                            deltas.front();
                    deltas.pop();
                }
                deltas.push(dCdA);
                dRelu = dReLU(this->layers[currLayer]->getNeuron(currNeuron)->getValue());

                gradientVec[gradientCounter++] = this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue() *
                                                 dRelu * dCdA;
            }
            gradientVec[gradientCounter++] = dCdA * dRelu;
        }
        currLayer--;
    }
    gradientList->push_back(gradientVec);
}

void NeuralNetwork::randomizeWeightsAndBiases() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    gen.seed(time(NULL));
    std::normal_distribution<float> d{0, 1};

    unsigned char currLayer = 1;
    unsigned char amountLayer = this->layers.size();

    while (currLayer < amountLayer) {
        int layerSize = this->layers[currLayer]->getSize();
        int prevLayerSize = this->layers[currLayer-1]->getSize();
        float scaleFactor = sqrtf(2.f / prevLayerSize);

        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float rValue = d(gen) * scaleFactor;
                this->layers[currLayer]->setWeight(rValue, prevNeuron, currNeuron);
            }
            this->layers[currLayer]->setBias(0, currNeuron);
        }

        currLayer++;
    }
}

void NeuralNetwork::updateWeightsAndBiases(float* negativeGradientVec) {

}

std::vector<Layer*> NeuralNetwork::getLayers() const {
    return this->layers;
}
