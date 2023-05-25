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

    netfile = new std::fstream;
}

NeuralNetwork::~NeuralNetwork() {
    unsigned char size = layers.size();
    for (int i = 0; i < size; i++) {
        delete layers[i];
    }
    layers.clear();

    delete netfile;
}

float* NeuralNetwork::getResults() const {
    unsigned char currLayer = this->layers.size() - 1;
    unsigned int layerSize = this->layers[currLayer]->getSize();
    float* endVec = new float[layerSize];

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        endVec[currNeuron] = this->layers[currLayer]->getNeuron(currNeuron)->getValue();
    }

    return endVec;
}

unsigned int NeuralNetwork::getGradientVecSize() const {
    unsigned int gradientSize = 0;
    unsigned char amountLayers = this->layers.size();

    for (int currLayer = 1; currLayer < amountLayers; currLayer++) {
        gradientSize += this->layers[currLayer-1]->getSize() *
                        this->layers[currLayer]->getSize() +
                        this->layers[currLayer]->getSize();
    }

    return gradientSize;
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

void NeuralNetwork::softmax(unsigned char layer) {
    unsigned int layerSize = this->layers[layer]->getSize();

    float highVal = this->layers[layer]->getNeuron(0)->getValue();
    for (int currNeuron = 1; currNeuron < layerSize; currNeuron++) {
        float curVal = this->layers[layer]->getNeuron(currNeuron)->getValue();
        if (curVal > highVal) { highVal = curVal; }
    }

    float total = 0;
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float curVal = this->layers[layer]->getNeuron(currNeuron)->getValue();
        this->layers[layer]->getNeuron(currNeuron)->setValue(curVal - highVal);
        total += expf(this->layers[layer]->getNeuron(currNeuron)->getValue());
    }

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float neuronVal = expf(this->layers[layer]->getNeuron(currNeuron)->getValue());
        this->layers[layer]->getNeuron(currNeuron)->setValue(neuronVal / total);
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

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float activationValue = 0;

        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            activationValue += this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue() * 
            this->layers[currLayer]->getWeight(prevNeuron, currNeuron);
        }

        activationValue += this->layers[currLayer]->getBias(currNeuron);
        this->layers[currLayer]->getNeuron(currNeuron)->setValue(activationValue);
    }

    softmax(currLayer);
}

void NeuralNetwork::backPropagate(float* correctData, float* gradientVec) {
    unsigned int gradientCounter = 0;

    unsigned char currLayer = this->layers.size() - 1;
    unsigned int currLayerSize = this->layers[currLayer]->getSize();
    unsigned int prevLayerSize = this->layers[currLayer-1]->getSize();
    std::vector<float> deltas;
    
    // Derivative of Cross-Entropy Loss (Multi-Class Classification)
    for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
        float softmax = this->layers[currLayer]->getNeuron(currNeuron)->getValue();
        float dSoftmax = softmax * (1 - softmax);
        float dCdA = - (correctData[currNeuron] / this->layers[currLayer]->getNeuron(currNeuron)->getValue());

        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            gradientVec[gradientCounter++] += dCdA * dSoftmax * this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue();
        }

        gradientVec[gradientCounter++] += dCdA * dSoftmax;
        deltas.push_back(dCdA * dSoftmax);
    }

    currLayer--;
    while (currLayer > 0) {
        currLayerSize = this->layers[currLayer]->getSize();
        prevLayerSize = this->layers[currLayer-1]->getSize();
        float nextLayerSize = this->layers[currLayer+1]->getSize();

        float deltaSum = 0;
        int deltasSize = deltas.size();
        for (int i = 0; i < deltasSize; i++) { deltaSum += deltas[i]; }
        deltas.clear();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            float dCdA = 0;
            for (int nextNeuron = 0; nextNeuron < nextLayerSize; nextNeuron++) {
                dCdA += this->layers[currLayer+1]->getWeight(currNeuron, nextNeuron) * deltaSum;
            }

            float dReLu = dReLU(this->layers[currLayer]->getNeuron(currNeuron)->getValue());

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                gradientVec[gradientCounter++] += dCdA * dReLu * this->layers[currLayer-1]->getNeuron(prevNeuron)->getValue();
            }

            gradientVec[gradientCounter++] += dCdA * dReLu;
            deltas.push_back(dCdA * dReLu);
        }
        currLayer--;
    }
}

void NeuralNetwork::initializeReLU() {
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

void NeuralNetwork::updateWeightsAndBiases(float* gradientVec) {
    unsigned int gradientSize = getGradientVecSize();
    unsigned char currLayer = this->layers.size() - 1;
    unsigned int gradientCounter = 0;

    float val = 0;
    while (currLayer > 0) {
        unsigned int currLayerSize = this->layers[currLayer]->getSize();
        unsigned int prevLayerSize = this->layers[currLayer-1]->getSize();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                val = this->layers[currLayer]->getWeight(prevNeuron, currNeuron) - gradientVec[gradientCounter++];
                this->layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
            }
            val = this->layers[currLayer]->getBias(currNeuron) - gradientVec[gradientCounter++];
            this->layers[currLayer]->setBias(val, currNeuron);
        }

        currLayer--;
    }
}

void NeuralNetwork::saveNetworkState(const char* path) {
    netfile->open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    
    int amountLayers = layers.size();
    /*netfile->write(reinterpret_cast<const char*>(&amountLayers), sizeof(int));

    for (int layer = 0; layer < amountLayers; layer++) {
        int layerSz = layers[layer]->getSize();
        netfile->write(reinterpret_cast<const char*>(&layerSz), sizeof(int));
    }*/

    int currLayer = 1;
    while (currLayer < amountLayers) {
        int currLayerSize = layers[currLayer]->getSize();

        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            int prevLayerSize = layers[currLayer-1]->getSize();

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float val = layers[currLayer]->getWeight(prevNeuron, currNeuron);
                netfile->write(reinterpret_cast<const char*>(&val), sizeof(float));
            }

            float val = layers[currLayer]->getBias(currNeuron);
            netfile->write(reinterpret_cast<const char*>(&val), sizeof(float));
        }

        currLayer++;
    }

    netfile->close();
}

void NeuralNetwork::loadNetworkState(const char* path) {
    netfile->open(path, std::ios::in | std::ios::binary);

    int amountLayers = layers.size();
    int currLayer = 1;

    while (currLayer < amountLayers) {
        int currLayerSize = layers[currLayer]->getSize();

        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            int prevLayerSize = layers[currLayer-1]->getSize();
            
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float val;
                netfile->read(reinterpret_cast<char*>(&val), sizeof(float));
                layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
            }

            float val;
            netfile->read(reinterpret_cast<char*>(&val), sizeof(float));
            layers[currLayer]->setBias(val, currNeuron);
        }

        currLayer++;
    }

    netfile->close();
}
