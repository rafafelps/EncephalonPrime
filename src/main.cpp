#include "Dataset.hpp"
#include "libattopng.hpp"
#include <iostream>

#define TRAINING 0

int main() {
    #if TRAINING
    Dataset data("dataset/training/train-labels.idx1-ubyte", "dataset/training/train-images.idx3-ubyte");
    #else
    Dataset data("dataset/test/t10k-labels.idx1-ubyte", "dataset/test/t10k-images.idx3-ubyte");
    #endif
    
    unsigned char img[28][28] = {0};
    int width = data.getWidth();
    int height = data.getHeight();
    int size = data.getSize();
    
    for (int amount = 0; amount < size; amount++) {
        libattopng_t* png = libattopng_new(28, 28, PNG_GRAYSCALE);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                data.images.read((char *)(&img[i][j]), sizeof(unsigned char));
                libattopng_set_pixel(png, j, i, img[i][j]);
            }
        }
        std::string s = std::to_string(amount+1);
        s = "images/" + s + ".png";
        libattopng_save(png, s.c_str());
        libattopng_destroy(png);
    }

    return 0;
}
