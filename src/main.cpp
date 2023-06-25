#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"

int main(int argc, char* argv[]) {
    Dataset data;
    if (argc == 2) {
        if (!atoi(argv[1])) {
            data.setData("dataset/training/train-labels.idx1-ubyte", "dataset/training/train-images.idx3-ubyte");
        } else {
            data.setData("dataset/test/t10k-labels.idx1-ubyte", "dataset/test/t10k-images.idx3-ubyte");
        }
    } else {
        std::cout << "Usage: .\\build.exe [0|1] (0: training; 1: test)" << std::endl;
        exit(1);
    }
    if (!data.getHeight() || !data.getWidth() || !data.getSize()) { std::cerr << "Failed to initialize data." << std::endl; exit(1); }

    unsigned int inputSize = 784;
    unsigned int outputSize = 10;
    std::vector<unsigned int> sizes;
    sizes.push_back(inputSize);
    sizes.push_back(16);
    sizes.push_back(16);
    sizes.push_back(outputSize);

    NeuralNetwork mnist(sizes);
    mnist.setDataset(&data);
    mnist.setName("mnist");

    //mnist.learn(1, false);
    mnist.loadNetworkState();

    unsigned int pos = 0;
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

    delete[] lastLayer;
    
    return 0;
}
