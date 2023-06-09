#pragma once
#include <fstream>
#include <string>

class Dataset {
private:
    unsigned int width;
    unsigned int height;
    unsigned int size;

    std::ifstream* label;
    std::ifstream* images;

    std::string pathLabel;
    std::string pathImages;
public:

    Dataset(std::string pathL, std::string pathI);
    Dataset();
    ~Dataset();

    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getSize() const;
    std::ifstream* getImages() const;
    std::ifstream* getLabel() const;
    std::string getPathLabel() const;
    std::string getPathImages() const;

    void setData(std::string pathL, std::string pathI);
};
