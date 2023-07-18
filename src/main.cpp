#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"

int main(int argc, char* argv[]) {
    Dataset trainingData;
    Dataset testData;
    trainingData.setData("dataset/training/train-labels.idx1-ubyte", "dataset/training/train-images.idx3-ubyte");
    testData.setData("dataset/test/t10k-labels.idx1-ubyte", "dataset/test/t10k-images.idx3-ubyte");
    if (!trainingData.getHeight() || !trainingData.getWidth() || !trainingData.getSize() ||
        !testData.getHeight() || !testData.getWidth() || !testData.getSize()) { std::cerr << "Failed to initialize data." << std::endl; exit(1); }

    unsigned int inputSize = 784;
    unsigned int outputSize = 10;
    std::vector<unsigned int> sizes;
    sizes.push_back(inputSize);
    sizes.push_back(16);
    sizes.push_back(16);
    sizes.push_back(outputSize);

    NeuralNetwork mnist(sizes);
    mnist.setName("mnist");

    if (atoi(argv[1])) {
        mnist.setDataset(&testData);
        mnist.loadNetworkState();
    } else {
        mnist.setDataset(&trainingData);
        mnist.learn(1, false);
    }

    unsigned int dataSize = testData.getSize();
    unsigned int correctPredictions = 0;
    for (unsigned int image = 0; image < dataSize; image++) {
        unsigned int label = testData.img[image]->label;
        mnist.propagate(testData.img[image]->values);

        float* lastLayer = mnist.getResults();
        unsigned int highVal = 0;
        for (unsigned int i = 0; i < outputSize; i++) {
            if (lastLayer[i] > lastLayer[highVal]) { highVal = i; }
        }

        if (label == highVal) { correctPredictions++; }

        delete[] lastLayer;
    }

    float acc = correctPredictions / static_cast<float>(dataSize);
    std::cout << "Accuracy: " << acc << std::endl;

    if (acc > atof(argv[2])) {
        std::cout << "Higher accuracy found!" << std::endl;
        return 0;
    }
}
