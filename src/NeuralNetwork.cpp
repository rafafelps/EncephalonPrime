#include <ctime>
#include <cmath>
#include <queue>
#include <random>
#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(std::string name) {
    this->name = name;

    std::ifstream netfile;
    netfile.open("bin/" + name + ".cfg", std::ios::in | std::ios::binary);
    
    unsigned int amountLayers;
    netfile.read(reinterpret_cast<char*>(&amountLayers), sizeof(unsigned int));
    
    unsigned int size;
    netfile.read(reinterpret_cast<char*>(&size), sizeof(unsigned int));
    
    layers.push_back(new Layer(size, NULL));
    for (int layer = 1; layer < amountLayers; layer++) {
        netfile.read(reinterpret_cast<char*>(&size), sizeof(unsigned int));
        layers.push_back(new Layer(size, layers[layer-1]));
    }

    netfile.close();
}

NeuralNetwork::NeuralNetwork(unsigned int layerAmount, unsigned int* sizes) {
    layers.push_back(new Layer(sizes[0], NULL));
    for (int layer = 1; layer < layerAmount; layer++) {
        layers.push_back(new Layer(sizes[layer], layers[layer-1]));
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
    unsigned char currLayer = layers.size() - 1;
    unsigned int layerSize = layers[currLayer]->getSize();
    float* endVec = new float[layerSize];

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        endVec[currNeuron] = layers[currLayer]->getNeuron(currNeuron)->getValue();
    }

    return endVec;
}

unsigned int NeuralNetwork::getGradientVecSize() const {
    unsigned int gradientSize = 0;
    unsigned char amountLayers = layers.size();

    for (int currLayer = 1; currLayer < amountLayers; currLayer++) {
        gradientSize += layers[currLayer-1]->getSize() *
                        layers[currLayer]->getSize() +
                        layers[currLayer]->getSize();
    }

    return gradientSize;
}

void NeuralNetwork::setDataset(Dataset* dataset) {
    this->dataset = dataset;
}

void NeuralNetwork::setName(std::string name) {
    this->name = name;
}

float NeuralNetwork::ReLU(float val) {
    return (val > 0) ? val : 0;
}

float NeuralNetwork::dReLU(float val) {
    return (val > 0) ? 1 : 0;
}

void NeuralNetwork::softmax(unsigned char layer) {
    unsigned int layerSize = layers[layer]->getSize();

    float highVal = layers[layer]->getNeuron(0)->getValue();
    for (int currNeuron = 1; currNeuron < layerSize; currNeuron++) {
        float curVal = layers[layer]->getNeuron(currNeuron)->getValue();
        if (curVal > highVal) { highVal = curVal; }
    }

    float total = 0;
    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float curVal = layers[layer]->getNeuron(currNeuron)->getValue();
        layers[layer]->getNeuron(currNeuron)->setValue(curVal - highVal);
        total += expf(layers[layer]->getNeuron(currNeuron)->getValue());
    }

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float neuronVal = expf(layers[layer]->getNeuron(currNeuron)->getValue());
        layers[layer]->getNeuron(currNeuron)->setValue(neuronVal / total);
    }
}

void NeuralNetwork::propagate(float* inputData) {
    unsigned char currLayer = 0;
    unsigned char amountLayers = layers.size() - 1;
    unsigned int layerSize = layers[currLayer]->getSize();

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        layers[currLayer]->getNeuron(currNeuron)->setValue(inputData[currNeuron]);
    }
    currLayer++;

    while (currLayer < amountLayers) {
        layerSize = layers[currLayer]->getSize();
        int prevLayerSize = layers[currLayer-1]->getSize();

        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            float activationValue = 0;

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                activationValue += layers[currLayer-1]->getNeuron(prevNeuron)->getValue() * 
                                   layers[currLayer]->getWeight(prevNeuron, currNeuron);
            }

            activationValue += layers[currLayer]->getBias(currNeuron);
            layers[currLayer]->getNeuron(currNeuron)->setValue(ReLU(activationValue));
        }
        currLayer++;
    }

    layerSize = layers[currLayer]->getSize();
    int prevLayerSize = layers[currLayer-1]->getSize();

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float activationValue = 0;

        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            activationValue += layers[currLayer-1]->getNeuron(prevNeuron)->getValue() * 
                               layers[currLayer]->getWeight(prevNeuron, currNeuron);
        }

        activationValue += layers[currLayer]->getBias(currNeuron);
        layers[currLayer]->getNeuron(currNeuron)->setValue(activationValue);
    }

    softmax(currLayer);
}

void NeuralNetwork::backPropagate(float* correctData, float* gradientVec) {
    unsigned int gradientCounter = 0;

    unsigned char currLayer = layers.size() - 1;
    unsigned int currLayerSize = layers[currLayer]->getSize();
    unsigned int prevLayerSize = layers[currLayer-1]->getSize();
    std::vector<float> deltas;
    
    // Derivative of Cross-Entropy Loss (Multi-Class Classification)
    for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
        float softmax = layers[currLayer]->getNeuron(currNeuron)->getValue();
        float dSoftmax = softmax * (1 - softmax);
        float dCdA = - (correctData[currNeuron] / layers[currLayer]->getNeuron(currNeuron)->getValue());

        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            gradientVec[gradientCounter++] += dCdA * dSoftmax * layers[currLayer-1]->getNeuron(prevNeuron)->getValue();
        }

        gradientVec[gradientCounter++] += dCdA * dSoftmax;
        deltas.push_back(dCdA * dSoftmax);
    }

    currLayer--;
    while (currLayer > 0) {
        currLayerSize = layers[currLayer]->getSize();
        prevLayerSize = layers[currLayer-1]->getSize();
        float nextLayerSize = layers[currLayer+1]->getSize();

        float deltaSum = 0;
        int deltasSize = deltas.size();
        for (int i = 0; i < deltasSize; i++) { deltaSum += deltas[i]; }
        deltas.clear();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            float dCdA = 0;
            for (int nextNeuron = 0; nextNeuron < nextLayerSize; nextNeuron++) {
                dCdA += layers[currLayer+1]->getWeight(currNeuron, nextNeuron) * deltaSum;
            }

            float dReLu = dReLU(layers[currLayer]->getNeuron(currNeuron)->getValue());

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                gradientVec[gradientCounter++] += dCdA * dReLu * layers[currLayer-1]->getNeuron(prevNeuron)->getValue();
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
    unsigned char amountLayer = layers.size();

    while (currLayer < amountLayer) {
        int layerSize = layers[currLayer]->getSize();
        int prevLayerSize = layers[currLayer-1]->getSize();
        float scaleFactor = sqrtf(2.f / prevLayerSize);

        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float rValue = d(gen) * scaleFactor;
                layers[currLayer]->setWeight(rValue, prevNeuron, currNeuron);
            }
            layers[currLayer]->setBias(0, currNeuron);
        }

        currLayer++;
    }
}

void NeuralNetwork::updateWeightsAndBiases(float* gradientVec) {
    unsigned int gradientSize = getGradientVecSize();
    unsigned char currLayer = layers.size() - 1;
    unsigned int gradientCounter = 0;

    float val = 0;
    while (currLayer > 0) {
        unsigned int currLayerSize = layers[currLayer]->getSize();
        unsigned int prevLayerSize = layers[currLayer-1]->getSize();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                val = layers[currLayer]->getWeight(prevNeuron, currNeuron) - gradientVec[gradientCounter++];
                layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
            }
            val = layers[currLayer]->getBias(currNeuron) - gradientVec[gradientCounter++];
            layers[currLayer]->setBias(val, currNeuron);
        }

        currLayer--;
    }
}

void NeuralNetwork::saveNetworkState() {
    std::ofstream netfile;
    netfile.open("bin/" + name + ".bin", std::ios::out | std::ios::binary | std::ios::trunc);
    
    int amountLayers = layers.size();
    int currLayer = 1;

    while (currLayer < amountLayers) {
        int currLayerSize = layers[currLayer]->getSize();

        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            int prevLayerSize = layers[currLayer-1]->getSize();

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float val = layers[currLayer]->getWeight(prevNeuron, currNeuron);
                netfile.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }

            float val = layers[currLayer]->getBias(currNeuron);
            netfile.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }

        currLayer++;
    }

    netfile.close();
    netfile.open("bin/" + name + ".cfg", std::ios::out | std::ios::binary | std::ios::trunc);

    netfile.write(reinterpret_cast<const char*>(&amountLayers), sizeof(unsigned int));

    for (int layer = 0; layer < amountLayers; layer++) {
        unsigned int layerSz = layers[layer]->getSize();
        netfile.write(reinterpret_cast<const char*>(&layerSz), sizeof(unsigned int));
    }
}

void NeuralNetwork::loadNetworkState() {
    std::ifstream netfile;
    netfile.open("bin/" + name + ".bin", std::ios::in | std::ios::binary);

    int amountLayers = layers.size();
    int currLayer = 1;

    while (currLayer < amountLayers) {
        int currLayerSize = layers[currLayer]->getSize();

        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            int prevLayerSize = layers[currLayer-1]->getSize();
            
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float val;
                netfile.read(reinterpret_cast<char*>(&val), sizeof(float));
                layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
            }

            float val;
            netfile.read(reinterpret_cast<char*>(&val), sizeof(float));
            layers[currLayer]->setBias(val, currNeuron);
        }

        currLayer++;
    }

    netfile.close();
}
