#include "Dataset.hpp"

#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))

Dataset::Dataset(std::string pathL, std::string pathI) {
    label = nullptr;
    images = nullptr;
    img = nullptr;
    setData(pathL, pathI);
}

Dataset::Dataset() {
    label = nullptr;
    images = nullptr;
    img = nullptr;
}

Dataset::~Dataset() {
    label->close();
    images->close();

    delete label;
    delete images;
    delete[] img;

    label = nullptr;
    images = nullptr;
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

std::ifstream* Dataset::getImages() const {
    return images;
}

std::ifstream* Dataset::getLabel() const {
    return label;
}

std::string Dataset::getPathLabel() const {
    return pathLabel;
}

std::string Dataset::getPathImages() const {
    return pathImages;
}

void Dataset::setData(std::string pathL, std::string pathI) {
    if (label || images) { return; }

    unsigned int valueL = 0;
    unsigned int valueI = 0;

    pathLabel = pathL;
    pathImages = pathI;

    label = new std::ifstream(pathL, std::ios::binary);
    images = new std::ifstream(pathI, std::ios::binary);

    label->read((char*)&valueL, 4);
    images->read((char*)&valueI, 4);
    valueL = SWAP_INT32(valueL);
    valueI = SWAP_INT32(valueI);
    if (valueL != 2049 || valueI != 2051) { this->~Dataset(); return; }

    label->read((char*)&valueL, 4);
    images->read((char*)&valueI, 4);
    valueL = SWAP_INT32(valueL);
    valueI = SWAP_INT32(valueI);
    if (valueL != valueI) { this->~Dataset(); return; }
    this->size = valueL;

    images->read((char*)&valueI, 4);
    valueI = SWAP_INT32(valueI);;
    this->height = valueI;

    images->read((char*)&valueI, 4);
    valueI = SWAP_INT32(valueI);
    this->width = valueI;

    images->seekg(16);
    label->seekg(8);
    img = new Image[size];
    for (int k = 0; k < size; k++) {
        img[k] = Image(width*height);
        
        unsigned char tmp;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                images->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
                img[k].values[i * width + j] = tmp / 255;
            }
        }
        label->read(reinterpret_cast<char*>(&tmp), sizeof(unsigned char));
        img[k].label = tmp;
    }
}
