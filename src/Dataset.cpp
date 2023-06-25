#include <fstream>
#include "Dataset.hpp"

#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))

Dataset::Dataset(std::string pathL, std::string pathI) {
    width = 0;
    height = 0;
    size = 0;
    img = nullptr;
    setData(pathL, pathI);
}

Dataset::Dataset() {
    width = 0;
    height = 0;
    size = 0;
    img = nullptr;
}

Dataset::~Dataset() {
    for (int i = 0; i < size; i++) {
        delete img[i];
    }
    delete[] img;

    width = 0;
    height = 0;
    size = 0;
    img = nullptr;
}

unsigned int Dataset::getWidth() const {
    return width;
}

unsigned int Dataset::getHeight() const {
    return height;
}

unsigned int Dataset::getSize() const {
    return size;
}

void Dataset::setData(std::string pathL, std::string pathI) {
    if (size || width || height) { return; }

    unsigned int valueL = 0;
    unsigned int valueI = 0;

    std::ifstream* label = new std::ifstream(pathL, std::ios::binary);
    std::ifstream* images = new std::ifstream(pathI, std::ios::binary);

    label->read(reinterpret_cast<char*>(&valueL), 4);
    images->read(reinterpret_cast<char*>(&valueI), 4);
    valueL = SWAP_INT32(valueL);
    valueI = SWAP_INT32(valueI);
    if (valueL != 2049 || valueI != 2051) { this->~Dataset(); return; }

    label->read(reinterpret_cast<char*>(&valueL), 4);
    images->read(reinterpret_cast<char*>(&valueI), 4);
    valueL = SWAP_INT32(valueL);
    valueI = SWAP_INT32(valueI);
    if (valueL != valueI) { this->~Dataset(); return; }
    this->size = valueL;

    images->read(reinterpret_cast<char*>(&valueI), 4);
    valueI = SWAP_INT32(valueI);;
    this->height = valueI;

    images->read(reinterpret_cast<char*>(&valueI), 4);
    valueI = SWAP_INT32(valueI);
    this->width = valueI;

    images->seekg(16);
    label->seekg(8);
    img = new Image*[size];
    for (int k = 0; k < size; k++) {
        img[k] = new Image(width*height);
        
        unsigned char tmp;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                images->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
                img[k]->values[i * width + j] = tmp / static_cast<float>(255);
            }
        }

        label->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
        img[k]->label = tmp;
    }

    label->close();
    images->close();
    delete label;
    delete images;
}
