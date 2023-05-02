#include "Dataset.hpp"

void littleEndianToBigEndian(unsigned int* value);

Dataset::Dataset(const char* pathL, const char* pathI) :
label(pathL, std::ios::binary),
images(pathI, std::ios::binary) {
    unsigned int valueL = 0;
    unsigned int valueI = 0;

    label.read((char*)&valueL, 4);
    images.read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueL);
    littleEndianToBigEndian(&valueI);
    if (valueL != 2049 || valueI != 2051) { this->~Dataset(); return; }

    valueL = 0; valueI = 0;
    label.read((char*)&valueL, 4);
    images.read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueL);
    littleEndianToBigEndian(&valueI);
    if (valueL != valueI) { this->~Dataset(); return; }
    this->size = valueL;

    valueI = 0;
    images.read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueI);
    this->height = valueI;

    valueI = 0;
    images.read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueI);
    this->width = valueI;
}

Dataset::~Dataset() {
    label.close();
    images.close();
}

unsigned int Dataset::getWidth() const {
    return this->width;
}

unsigned int Dataset::getHeight() const {
    return this->height;
}

unsigned int Dataset::getSize() const {
    return this->size;
}

void littleEndianToBigEndian(unsigned int* value) {
    unsigned int b1 = 0xFF000000;
    unsigned int tmp = *value;

    *value = 0;
    *value += (tmp & b1) >> 24;
    *value += (tmp & (b1 >> 8)) >> 8;
    *value += (tmp & (b1 >> 16)) << 8;
    *value += (tmp & (b1 >> 24)) << 24;
}
