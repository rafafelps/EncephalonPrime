#pragma once

class Neuron {
private:
    float value;
    float actValue;
public:
    Neuron(float value = 0);
    ~Neuron();

    float getValue() const;
    float getActValue() const;
    void setValue(float value);
    void setActValue(float value);
};
