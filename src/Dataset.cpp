#include "Dataset.hpp"

void littleEndianToBigEndian(unsigned int* value);

Dataset::Dataset(const char* pathL, const char* pathI) {
    this->images = NULL;
    this->label = NULL;
    setData(pathL, pathI);
}

Dataset::Dataset() {}

Dataset::~Dataset() {
    label->close();
    images->close();

    delete label;
    delete images;
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

std::ifstream* Dataset::getImages() const {
    return this->images;
}

std::ifstream* Dataset::getLabel() const {
    return this->label;
}

void Dataset::setData(const char* pathL, const char* pathI) {
    unsigned int valueL = 0;
    unsigned int valueI = 0;

    if (!this->images && !this->label) {
        label = new std::ifstream(pathL, std::ios::binary);
        images = new std::ifstream(pathI, std::ios::binary);
    } else {
        exit(static_cast<int>('P'));
    }

    label->read((char*)&valueL, 4);
    images->read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueL);
    littleEndianToBigEndian(&valueI);
    if (valueL != 2049 || valueI != 2051) { this->~Dataset(); return; }

    valueL = 0; valueI = 0;
    label->read((char*)&valueL, 4);
    images->read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueL);
    littleEndianToBigEndian(&valueI);
    if (valueL != valueI) { this->~Dataset(); return; }
    this->size = valueL;

    valueI = 0;
    images->read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueI);
    this->height = valueI;

    valueI = 0;
    images->read((char*)&valueI, 4);
    littleEndianToBigEndian(&valueI);
    this->width = valueI;
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
