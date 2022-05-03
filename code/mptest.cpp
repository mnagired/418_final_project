
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

// Number of testing samples
const int nTesting = 10000;

// Image size in MNIST database
const int width  = 28;
const int height = 28;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

template<typename Type>
std::vector<Type> vector(const std::size_t __n, Type value = Type()) {
    return std::vector<Type>(__n, value);
}
template<typename Type>
std::vector<std::vector<Type>>
matrix(const std::size_t __n, const std::size_t __m, Type value = Type()) {
    return std::vector<std::vector<Type>>(__n, std::vector<Type>(__m, value));
}

class Network {
public:
    struct Layer {
        const std::size_t in;
        const std::size_t out;

        std::vector<std::vector<double>> weights;

        Layer(const std::size_t __in, const std::size_t __out)
            : in(__in),
              out(__out),
              weights(matrix<double>(__in, __out, 0.0)) { }
    };

    const std::size_t samples = 50000ul;
    const std::size_t input   = 784ul;
    const std::size_t hidden  = 128ul;
    const std::size_t output  = 10ul;

private:
    std::ifstream      image;
    std::ifstream      label;
    std::vector<Layer> layers;

    std::vector<double> out1;
    std::vector<double> in2;
    std::vector<double> out2;
    std::vector<double> in3;
    std::vector<double> out3;
    std::vector<double> expected;

    std::vector<std::vector<double>> data;

    std::size_t __read() {
        // Reading image
        std::uint8_t __tmp;
        for (std::size_t j = 1; j <= height; ++j) {
            for (std::size_t i = 1; i <= width; ++i) {
                image.read((char*)&__tmp, sizeof(std::uint8_t));
                switch (__tmp) {
                    case 0:
                        this->data[i][j] = 0;
                        break;
                    default:
                        this->data[i][j] = 1;
                        break;
                }
            }
        }

        for (std::size_t j = 1; j <= height; ++j) {
            for (std::size_t i = 1; i <= width; ++i) {
                std::size_t pos = i + (j - 1) * width;
                out1[pos]       = this->data[i][j];
            }
        }

        // Reading label
        label.read((char*)&__tmp, sizeof(char));
        for (int i = 1; i <= this->output; ++i) { expected[i] = 0.0; }
        expected[__tmp + 1] = 1.0;

        return (std::size_t)(__tmp);
    }

    void __perceptron() {
        for (std::size_t i = 1; i <= this->hidden; ++i) { this->in2[i] = 0.0; }

        for (std::size_t i = 1; i <= this->output; ++i) { this->in3[i] = 0.0; }

        for (std::size_t i = 1; i <= this->input; ++i) {
            for (std::size_t j = 1; j <= this->hidden; ++j) {
                this->in2[j] +=
                    this->out1[i] * this->layers.at(0).weights[i][j];
            }
        }

        for (std::size_t i = 1; i <= this->hidden; ++i) {
            this->out2[i] = sigmoid(this->in2[i]);
        }

        for (std::size_t i = 1; i <= this->hidden; ++i) {
            for (std::size_t j = 1; j <= this->output; ++j) {
                this->in3[j] +=
                    this->out2[i] * this->layers.at(1).weights[i][j];
            }
        }

        for (std::size_t i = 1; i <= this->output; ++i) {
            this->out3[i] = sigmoid(in3[i]);
        }
    }

    double __error() {
        double res = 0.0;
        for (std::size_t i = 1; i <= this->output; ++i) {
            res += (this->out3[i] - this->expected[i]) *
                   (this->out3[i] - this->expected[i]);
        }
        res *= 0.5;
        return res;
    }

public:
    Network() {
        this->layers.emplace_back(this->input + 1ul, this->hidden + 1ul);
        this->layers.emplace_back(this->hidden + 1ul, this->output + 1ul);

        this->data = matrix<double>(29ul, 29ul);

        this->out1     = vector<double>(this->input + 1);
        this->in2      = vector<double>(this->hidden + 1);
        this->out2     = vector<double>(this->hidden + 1);
        this->in3      = vector<double>(this->output + 1);
        this->out3     = vector<double>(this->output + 1);
        this->expected = vector<double>(this->output + 1);
    }

    void info() {
        std::cout << ">>>>> C++ OPENMP MNIST INFORMATION <<<<<" << std::endl;
        std::cout << "  >>> LAYERS" << std::endl;
        std::cout << "    > Input: " << this->input << std::endl;
        std::cout << "    > Hidden: " << this->hidden << std::endl;
        std::cout << "    > Output: " << this->output << std::endl;
        std::cout << std::endl;
    }

    void load() {
        std::cout << ">>> LOADING WEIGHTS AND TEST DATA" << std::endl;
        std::ifstream file("model-neural-network.dat", std::ios::in);
        if (!file.is_open()) { return; }

        // Input layer - Hidden layer
        for (std::size_t i = 1ul; i <= this->input; ++i) {
            for (std::size_t j = 1ul; j <= this->hidden; ++j) {
                file >> this->layers.at(0).weights[i][j];
            }
        }

        // Hidden layer - Output layer
        for (std::size_t i = 1ul; i <= this->hidden; ++i) {
            for (std::size_t j = 1ul; j <= this->output; ++j) {
                file >> this->layers.at(1).weights[i][j];
            }
        }

        file.close();

        this->image.open("mnist/t10k-images-idx3-ubyte",
                         std::ios::in | std::ios::binary);
        this->label.open("mnist/t10k-labels-idx1-ubyte",
                         std::ios::in | std::ios::binary);
        std::uint8_t __tmp;
        for (std::size_t i = 1ul; i <= 16ul; ++i) {
            image.read((char*)&__tmp, sizeof(std::uint8_t));
        }
        for (std::size_t i = 1ul; i <= 8ul; ++i) {
            label.read((char*)&__tmp, sizeof(std::uint8_t));
        }
    }

    void test() {
        std::cout << ">>> RUNNING TEST LOOP" << std::endl;
        int nCorrect = 0;
        for (int sample = 1; sample <= nTesting; ++sample) {
            // std::cout << "Sample " << sample << std::endl;

            // Getting (image, label)
            auto label = this->__read();

            // Classification - Perceptron procedure
            this->__perceptron();

            // Prediction
            std::size_t predict = 1;
            for (std::size_t i = 2; i <= this->output; ++i) {
                if (this->out3[i] > this->out3[predict]) { predict = i; }
            }
            --predict;

            // Write down the classification result and the squared error
            // double error = this->__error();
            // printf("Error: %0.6lf\n", error);

            if (label == predict) {
                ++nCorrect;
                //     std::cout << "Sample " << sample << ": YES. Label = " <<
                //     label
                //               << ". Predict = " << predict << ". Error = " <<
                //               error
                //               << std::endl;
            }
            // } else {
            //     std::cout << "Sample " << sample << ": NO.  Label = " <<
            //     label
            //               << ". Predict = " << predict << ". Error = " <<
            //               error
            //               << std::endl;
            // }
        }

        double accuracy = (double)(nCorrect) / nTesting * 100.0;

        std::cout << "Number of correct samples: " << nCorrect << " / "
                  << nTesting << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;
    }

    void cleanup() {
        this->image.close();
        this->label.close();
    }
};

int main(int argc, char** argv) {
    auto network = Network();
    network.info();

    // Neural Network Initialization
    network.load();

    network.test();

    network.cleanup();

    return 0;
}