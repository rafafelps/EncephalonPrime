#pragma once
#include <fstream>

class Dataset {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;
public:
    std::ifstream label;
    std::ifstream images;

    Dataset(const char* pathL, const char* pathI);
    ~Dataset();
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getSize() const;
};
