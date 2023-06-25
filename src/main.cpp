#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"
#include "libattopng.hpp"

int getImage(float* inputData, Dataset* data);
float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset* data, NeuralNetwork* net);

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
    if (!data.getLabel() || !data.getImages()) { std::cerr << "Failed to initialize data." << std::endl; exit(1); }

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
    mnist.loadNetworkState();

    int label = 0;
    float* inputData = new float[inputSize];

    int width = data.getWidth();
    int height = data.getHeight();
    int dataSize = data.getSize();

    std::random_device rd{};
    std::mt19937 gen(rd());
    gen.seed(time(NULL));
    std::vector<unsigned int> imageOrder;
    for (int i = 0; i < dataSize; i++) { imageOrder.push_back(i); }

    unsigned int gradientSize = mnist.getGradientVecSize();
    float* gradientVec = new float[gradientSize]();
    float* correctData = new float[outputSize]();

    unsigned int t = 0;
    float* m = new float[gradientSize]();
    float* v = new float[gradientSize]();
    /*for (unsigned int epoch = 0; epoch < 1; epoch++) {
        data.getImages()->seekg(16);
        data.getLabel()->seekg(8);
        std::shuffle(imageOrder.begin(), imageOrder.end(), gen);
        unsigned int totalEval = 0;
        unsigned int correctEval = 0;

        for (unsigned int image = 0; image < dataSize; image++) {
            t++;
            label = data.img[imageOrder[image]]->label;
            correctData[label]++;

            mnist.propagate(data.img[imageOrder[image]]->values);
            mnist.adam(t, correctData, m, v);

            float* lastLayer = mnist.getResults();
            unsigned int highVal = 0;
            for (int i = 0; i < outputSize; i++) {
                if (lastLayer[i] > lastLayer[highVal]) {
                    highVal = i;
                }
            }
            if (highVal == label) { correctEval++; }
            totalEval++;

            float acc = static_cast<float>(correctEval) / totalEval;
            if (acc >= 0.84) {
                mnist.saveNetworkState();
                std::cout << std::endl << "Finished learning!" << std::endl;
                return 0;
            }

            std::cout << "\rImage: " << image << "  Accuracy: " << acc << std::flush;

            correctData[label]--;
        }
    }
    mnist.saveNetworkState();*/

    data.getImages()->seekg(16);
    data.getLabel()->seekg(8);

    unsigned int pos = 0;
    label = data.img[pos]->label;
    mnist.propagate(data.img[pos]->values);

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
    delete[] inputData;
    delete[] gradientVec;
    delete[] correctData;
    delete[] m;
    delete[] v;
    
    return 0;
}

int getImage(float* inputData, Dataset* data) {
    unsigned int width = data->getWidth();
    unsigned int height = data->getHeight();

    unsigned char tmp;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data->getImages()->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
            inputData[i * width + j] = tmp / 255;
        }
    }
    data->getLabel()->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
    return tmp;
}

float* trainPartition(unsigned int partition, unsigned int partitionSize, Dataset* data, NeuralNetwork* net) {
    Dataset* new_data = new Dataset(data->getPathLabel(), data->getPathImages());
    NeuralNetwork* new_net = new NeuralNetwork(net);
    new_net->setDataset(new_data);
    
    unsigned int startPos = partition * partitionSize;
    new_data->getLabel()->seekg(8 + startPos);
    new_data->getImages()->seekg(16 + (startPos * new_data->getWidth() * new_data->getHeight()));

    unsigned int gradientVecSize = new_net->getGradientVecSize();
    float* gradientVec = new float[gradientVecSize]();
    float* fInputData = new float[new_data->getWidth() * new_data->getHeight()];
    float* correctData = new float[gradientVecSize]();
    
    int dataSize = startPos + partitionSize;
    if (startPos + (2 * partitionSize) > new_data->getSize()) {
        dataSize = new_data->getSize();
    }
    
    int amountRun = 0;
    for (int i = startPos; i < dataSize; i++, amountRun++) {        
        int label = getImage(fInputData, new_data);

        correctData[label]++;
        new_net->propagate(fInputData);
        new_net->backPropagate(correctData, gradientVec);
        correctData[label]--;
    }

    for (int i = 0; i < gradientVecSize; i++) { gradientVec[i] /= amountRun; }

    delete new_data;
    delete new_net;
    delete[] correctData;
    delete[] fInputData;

    return gradientVec;
}
