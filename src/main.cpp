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

        if (label != highVal) {
            continue;
            unsigned int width = testData.getWidth();
            unsigned int height = testData.getHeight();
            libattopng_t* png = libattopng_new(width, height, PNG_GRAYSCALE);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    libattopng_put_pixel(png, testData.img[image]->values[i * width + j] * 255);
                }
            }
            for (int i = 0; i < outputSize; i++) {
                if (label != i) {
                    std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "%" << std::endl;
                } else {
                    std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "% <-" << std::endl;
                }
            }
            libattopng_save(png, "images/result.png");
            libattopng_destroy(png);
        } else { correctPredictions++; }

        delete[] lastLayer;
    }

    std::cout << std::endl << "Accuracy: " << correctPredictions / static_cast<float>(dataSize) << std::endl;

    /*unsigned int pos = 3;
    unsigned int label = data.img[pos]->label;
    mnist.propagate(data.img[pos]->values);

    unsigned int width = data.getWidth();
    unsigned int height = data.getHeight();
    libattopng_t* png = libattopng_new(width, height, PNG_GRAYSCALE);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            libattopng_put_pixel(png, data.img[pos]->values[i * width + j] * 255);
        }
    }
    libattopng_save(png, "images/result.png");
    libattopng_destroy(png);

    float* lastLayer = mnist.getResults();
    for (int i = 0; i < outputSize; i++) {
        if (label != i) {
            std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "%" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(3) << i << ": " << lastLayer[i] * 100 << "% <-" << std::endl;
        }
    }

    delete[] lastLayer;*/
    
    return 0;
}
