#include "Dataset.hpp"
#include <iostream>

int main() {
    Dataset data("dataset/training/train-labels.idx1-ubyte", "dataset/training/train-images.idx3-ubyte");
    
    std::cout << data.getSize() << std::endl;

    return 0;
}
