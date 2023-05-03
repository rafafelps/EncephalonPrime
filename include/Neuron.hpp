#pragma once

class Neuron {
private:
    float value;
public:
    Neuron(float value = 0);
    ~Neuron();

    float getValue() const;
    void setValue(float value);
};
