#include "Dataset.hpp"
#include "libattopng.hpp"
#include <iostream>

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
        exit(69420);
    }
    
    unsigned char img[28][28] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();
    
    for (int amount = 0; amount < size; amount++) {
        libattopng_t* png = libattopng_new(28, 28, PNG_GRAYSCALE);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                data.getImages()->read((char *)(&img[i][j]), sizeof(unsigned char));
                libattopng_set_pixel(png, j, i, img[i][j]);
            }
        }
        std::string s = std::to_string(amount+1);
        s = "images/" + s + ".png";
        libattopng_save(png, s.c_str());
        libattopng_destroy(png);
        //std::cerr << "\rRemaining: " << size - amount - 1 << std::flush;
    }
    return 0;
}