#include <ctime>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <limits>
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

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> sizes) {
    unsigned int layerAmount = sizes.size();
    layers.push_back(new Layer(sizes[0], NULL));
    for (int layer = 1; layer < layerAmount; layer++) {
        layers.push_back(new Layer(sizes[layer], layers[layer-1]));
    }
}

NeuralNetwork::NeuralNetwork(NeuralNetwork* originalNet) {
    unsigned int layerAmount = originalNet->layers.size();
    layers.push_back(new Layer(originalNet->layers[0]->getSize(), NULL));
    for (int layer = 1; layer < layerAmount; layer++) {
        layers.push_back(new Layer(originalNet->layers[layer]->getSize(), originalNet->layers[layer-1]));
    }

    int currLayer = 1;

    while (currLayer < layerAmount) {
        int currLayerSize = layers[currLayer]->getSize();

        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            int prevLayerSize = layers[currLayer-1]->getSize();
            
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                float val = originalNet->layers[currLayer]->getWeight(prevNeuron, currNeuron);
                layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
            }

            float val = originalNet->layers[currLayer]->getBias(currNeuron);
            layers[currLayer]->setBias(val, currNeuron);
        }

        currLayer++;
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
        endVec[currNeuron] = layers[currLayer]->getNeuron(currNeuron)->getActValue();
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

float NeuralNetwork::getCost(unsigned int correctResult) const {
    return - logf(layers[layers.size() - 1]->getNeuron(correctResult)->getActValue() + 1e-8);
}

void NeuralNetwork::setDataset(Dataset* dataset) {
    this->dataset = dataset;
}

void NeuralNetwork::setName(std::string name) {
    this->name = name;
}

void NeuralNetwork::adam(unsigned int t, float* correctData, float* m, float* v, float alpha, float beta1, float beta2, float epsilon) {
    unsigned int gradVecSize = getGradientVecSize();
    float* gradientVec = new float[gradVecSize]();
    backPropagate(correctData, gradientVec);

    float* mhat = new float[gradVecSize];
    float* vhat = new float[gradVecSize];

    for (int i = 0; i < gradVecSize; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * gradientVec[i];
        v[i] = beta2 * v[i] + (1 - beta2) * (gradientVec[i] * gradientVec[i]);
        mhat[i] = m[i] / (1 - powf(beta1, t));
        vhat[i] = v[i] / (1 - powf(beta2, t));
    }

    unsigned char currLayer = layers.size() - 1;
    unsigned int gradientCounter = 0;

    float val = 0;
    while (currLayer > 0) {
        unsigned int currLayerSize = layers[currLayer]->getSize();
        unsigned int prevLayerSize = layers[currLayer-1]->getSize();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                val = layers[currLayer]->getWeight(prevNeuron, currNeuron) - alpha * (mhat[gradientCounter] / (sqrtf(vhat[gradientCounter]) + epsilon));
                layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
                gradientCounter++;
            }
            val = layers[currLayer]->getBias(currNeuron) - alpha * (mhat[gradientCounter] / (sqrtf(vhat[gradientCounter]) + epsilon));
            layers[currLayer]->setBias(val, currNeuron);
            gradientCounter++;
        }

        currLayer--;
    }

    delete[] gradientVec;
    delete[] mhat;
    delete[] vhat;
}

float NeuralNetwork::ReLU(float val) {
    return (val > 0) ? val : 0;
}

float NeuralNetwork::dReLU(float val) {
    return (val > 0) ? 1 : 0;
}

float NeuralNetwork::sigmoid(float val) {
    return 1 / (1 + expf(-val));
}

float NeuralNetwork::dSigmoid(float val) {
    float sigVal = sigmoid(val);
    return sigVal * (1 - sigVal);
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
        total += expf(curVal - highVal);
    }

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float curVal = layers[layer]->getNeuron(currNeuron)->getValue();
        float neuronVal = expf(curVal - highVal);
        layers[layer]->getNeuron(currNeuron)->setActValue(neuronVal / total);
    }
}

void NeuralNetwork::learn(unsigned int epochs, bool loadFromFile) {
    loadFromFile ? loadNetworkState() : kaimingInitialization();

    std::random_device rd{};
    std::mt19937 gen(rd());
    gen.seed(0);

    unsigned int dataSize = dataset->getSize();
    std::vector<unsigned int> imageOrder;
    for (int i = 0; i < dataSize; i++) { imageOrder.push_back(i); }

    unsigned int outputSize = layers[layers.size()-1]->getSize();
    unsigned int gradientSize = getGradientVecSize();
    float* m = new float[gradientSize]();
    float* v = new float[gradientSize]();

    unsigned int t = 0;
    float highAcc = 0;
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {
        unsigned int totalEval = 0;
        unsigned int correctEval = 0;
        unsigned int label;
        float* correctData = new float[outputSize]();
        std::shuffle(imageOrder.begin(), imageOrder.end(), gen);

        for (unsigned int image = 0; image < dataSize; image++) {
            t++;
            label = dataset->img[imageOrder[image]]->label;
            correctData[label]++;

            propagate(dataset->img[imageOrder[image]]->values);
            adam(t, correctData, m, v);

            float* lastLayer = getResults();
            unsigned int highVal = 0;
            for (int i = 0; i < outputSize; i++) {
                if (lastLayer[i] > lastLayer[highVal]) {
                    highVal = i;
                }
            }
            if (highVal == label) { correctEval++; }
            totalEval++;

            float acc = static_cast<float>(correctEval) / totalEval;
            if (acc > highAcc) { highAcc = acc; }
            else {
                float change = highAcc / (acc + std::numeric_limits<float>::min());
                if (image > 10000 && change > 1.0 && change < 1.002) {
                    saveNetworkState();
                }
            }
            /*if (acc >= 0.84) {
                saveNetworkState();
                std::cout << std::endl << "Finished learning!" << std::endl;
                break;
            }*/

            std::cout << "\rImage: " << image << "  Accuracy: " << acc << std::flush;

            correctData[label]--;
        }

        delete[] correctData;
    }
    saveNetworkState();
    
    delete[] m;
    delete[] v;
}

void NeuralNetwork::propagate(float* inputData) {
    unsigned char currLayer = 0;
    unsigned char amountLayers = layers.size() - 1;
    unsigned int layerSize = layers[currLayer]->getSize();

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        layers[currLayer]->getNeuron(currNeuron)->setActValue(inputData[currNeuron]);
    }
    currLayer++;

    while (currLayer < amountLayers) {
        layerSize = layers[currLayer]->getSize();
        int prevLayerSize = layers[currLayer-1]->getSize();

        for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
            float activationValue = 0;

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                activationValue += layers[currLayer-1]->getNeuron(prevNeuron)->getActValue() * 
                                   layers[currLayer]->getWeight(prevNeuron, currNeuron);
            }

            activationValue += layers[currLayer]->getBias(currNeuron);
            layers[currLayer]->getNeuron(currNeuron)->setValue(activationValue);
            layers[currLayer]->getNeuron(currNeuron)->setActValue(ReLU(activationValue));
        }
        currLayer++;
    }

    layerSize = layers[currLayer]->getSize();
    int prevLayerSize = layers[currLayer-1]->getSize();

    for (int currNeuron = 0; currNeuron < layerSize; currNeuron++) {
        float activationValue = 0;

        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            activationValue += layers[currLayer-1]->getNeuron(prevNeuron)->getActValue() * 
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
        float softVal = layers[currLayer]->getNeuron(currNeuron)->getActValue();
        float dAct = softVal * (1 - softVal);

        float dCdA = softVal - correctData[currNeuron];

        for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
            gradientVec[gradientCounter++] += dCdA * dAct * layers[currLayer-1]->getNeuron(prevNeuron)->getActValue();
        }

        gradientVec[gradientCounter++] += dCdA * dAct;
        deltas.push_back(dCdA * dAct);
    }

    currLayer--;
    while (currLayer > 0) {
        currLayerSize = layers[currLayer]->getSize();
        prevLayerSize = layers[currLayer-1]->getSize();
        unsigned int nextLayerSize = layers[currLayer+1]->getSize();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            float dCdA = 0;
            for (int nextNeuron = 0; nextNeuron < nextLayerSize; nextNeuron++) {
                dCdA += layers[currLayer+1]->getWeight(currNeuron, nextNeuron) * deltas[nextNeuron];
            }

            float dReLu = dReLU(layers[currLayer]->getNeuron(currNeuron)->getValue());

            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                gradientVec[gradientCounter++] += dCdA * dReLu * layers[currLayer-1]->getNeuron(prevNeuron)->getActValue();
            }

            gradientVec[gradientCounter++] += dCdA * dReLu;
            deltas.push_back(dCdA * dReLu);
        }

        deltas.erase(deltas.begin(), deltas.begin() + nextLayerSize);
        currLayer--;
    }
}

void NeuralNetwork::kaimingInitialization() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    gen.seed(0);
    std::normal_distribution<float> d{0, 1};

    unsigned char currLayer = 1;
    unsigned char amountLayer = layers.size();

    while (currLayer < amountLayer) {
        int layerSize = layers[currLayer]->getSize();
        int prevLayerSize = layers[currLayer-1]->getSize();
        float scaleFactor = sqrtf(2.0 / prevLayerSize);

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

void NeuralNetwork::updateWeightsAndBiases(float learningRate, float* gradientVec) {
    unsigned int gradientSize = getGradientVecSize();
    unsigned char currLayer = layers.size() - 1;
    unsigned int gradientCounter = 0;

    float val = 0;
    while (currLayer > 0) {
        unsigned int currLayerSize = layers[currLayer]->getSize();
        unsigned int prevLayerSize = layers[currLayer-1]->getSize();
        
        for (int currNeuron = 0; currNeuron < currLayerSize; currNeuron++) {
            for (int prevNeuron = 0; prevNeuron < prevLayerSize; prevNeuron++) {
                val = layers[currLayer]->getWeight(prevNeuron, currNeuron) - (learningRate * gradientVec[gradientCounter++]);
                layers[currLayer]->setWeight(val, prevNeuron, currNeuron);
            }
            val = layers[currLayer]->getBias(currNeuron) - (learningRate * gradientVec[gradientCounter++]);
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
