#pragma once

class Image {
public:
    unsigned int label;
    float* values;
    
    Image(unsigned int size) { values = new float[size]; }
    Image() { values = nullptr; }
    ~Image() { delete[] values; }
};
