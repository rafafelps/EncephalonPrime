#pragma once
#include <fstream>

class Dataset {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;

    std::ifstream* label;
    std::ifstream* images;
public:

    Dataset(const char* pathL, const char* pathI);
    Dataset();
    ~Dataset();

    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getSize() const;
    std::ifstream* getImages() const;
    std::ifstream* getLabel() const;

    void setData(const char* pathL, const char* pathI);
};
