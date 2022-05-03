
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <set>
#include <string>
#include <vector>

double sigmoid(double __x) {
    return 1.0 / (1.0 + std::exp(-__x));
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
    const std::size_t epochs  = 0ul;
    const std::size_t threads = 1ul;

    const double learning_rate = 1.0e-3;
    const double momentum      = 0.9;
    const double epsilon       = 1.0e-3;

private:
    struct Data {
        std::vector<std::vector<double>> image;
        std::vector<std::vector<double>> label;
    } data;
    std::vector<Layer> layers;

    void __read_images() {
        std::ifstream file;
        file.open("mnist/train-images-idx3-ubyte", std::ios::binary);
        if (!file.is_open()) { return; }
        const auto   __ratio = sizeof(std::uint32_t) / sizeof(std::uint8_t);
        std::uint8_t __tmp;

        std::uint32_t magic = 0;
        for (std::size_t i = 0; i < __ratio; ++i) {
            file.read((char*)&__tmp, sizeof(std::uint8_t));
            magic += __tmp << (0x8 * (__ratio - (i + 1)));
        }
        assert(magic == 0x00000803u);

        std::uint32_t num_images = 0;
        for (std::size_t i = 0; i < __ratio; ++i) {
            file.read((char*)&__tmp, sizeof(std::uint8_t));
            num_images += __tmp << (0x8 * (__ratio - (i + 1)));
        }
        assert(num_images == 60000u);

        std::uint32_t num_rows = 0;
        for (std::size_t i = 0; i < __ratio; ++i) {
            file.read((char*)&__tmp, sizeof(std::uint8_t));
            num_rows += __tmp << (0x8 * (__ratio - (i + 1)));
        }
        assert(num_rows == 28u);

        std::uint32_t num_cols = 0;
        for (std::size_t i = 0; i < __ratio; ++i) {
            file.read((char*)&__tmp, sizeof(std::uint8_t));
            num_cols += __tmp << (0x8 * (__ratio - (i + 1)));
        }
        assert(num_cols == 28u);

        this->data.image = matrix<double>(num_images, num_rows * num_cols);

        for (std::uint32_t i = 0; i < num_images; ++i) {
            for (std::uint32_t r = 0; r < num_rows; ++r) {
                for (std::uint32_t c = 0; c < num_cols; ++c) {
                    file.read((char*)&__tmp, sizeof(std::uint8_t));
                    switch (__tmp) {
                        case 0:
                            this->data.image[i][(num_rows * r) + c] = 0.0;
                            break;
                        default:
                            this->data.image[i][(num_rows * r) + c] = 1.0;
                            break;
                    }
                }
            }
        }
        file.close();
    }

    void __read_labels() {
        std::ifstream file("mnist/train-labels-idx1-ubyte", std::ios::binary);

        if (!file.is_open()) { return; }

        const auto   __ratio = sizeof(std::uint32_t) / sizeof(std::uint8_t);
        std::uint8_t __tmp;

        std::uint32_t magic = 0;
        for (std::size_t i = 0; i < __ratio; ++i) {
            file.read((char*)&__tmp, sizeof(std::uint8_t));
            magic += __tmp << (0x8 * (__ratio - (i + 1)));
        }
        assert(magic == 0x00000801u);

        std::uint32_t num_images = 0;
        for (std::size_t i = 0; i < __ratio; ++i) {
            file.read((char*)&__tmp, sizeof(std::uint8_t));
            num_images += __tmp << (0x8 * (__ratio - (i + 1)));
        }
        assert(num_images == 60000u);

        this->data.label = matrix<double>(num_images, 10ul);

        for (std::uint32_t i = 0; i < num_images; ++i) {

            file.read((char*)&__tmp, sizeof(std::uint8_t));

            for (std::size_t j = 0ul; j < this->output; ++j) {
                this->data.label[i][j] = 0.0;
            }
            this->data.label[i][std::size_t(__tmp)] = 1.0;
        }
    }

    double __backprop(const std::vector<std::vector<double>>& delta,
                      const std::vector<double>&              theta,
                      const std::vector<double>& out, const std::size_t i,
                      const std::size_t j) {
        auto learn  = this->learning_rate * theta[j] * out[i];
        auto moment = this->momentum * delta[i][j];
        return learn + moment;
    }

public:
    Network(const std::size_t threads, const std::size_t epochs,
            const double learning_rate) {
        this->layers.emplace_back(this->input, this->hidden);
        this->layers.emplace_back(this->hidden, this->output);
    }

    void info() {
        std::cout << ">>>>> C++ OPENMP MNIST INFORMATION <<<<<" << std::endl;
        std::cout << "  >>> LAYERS" << std::endl;
        std::cout << "    > Input: " << this->input << std::endl;
        std::cout << "    > Hidden: " << this->hidden << std::endl;
        std::cout << "    > Output: " << this->output << std::endl;
        std::cout << "  >>> HYPERPARAMETERS" << std::endl;
        std::cout << "    > Epochs: " << this->epochs << std::endl;
        std::cout << "    > Learning rate: " << this->learning_rate
                  << std::endl;
        std::cout << "    > Momentum: " << this->momentum << std::endl;
        std::cout << "    > Epsilon: " << this->epsilon << std::endl;
        std::cout << std::endl;
    }

    void read_data() {
        this->__read_images();
        this->__read_labels();
    }

    void train() {
        const auto sample_per_thread = this->samples / threads;
#pragma omp parallel num_threads(threads) default(none)                        \
    shared(std::cout, this->data, this->layers, sample_per_thread)
        {
            auto w1     = matrix<double>(this->input, this->hidden);
            auto delta1 = matrix<double>(this->input, this->hidden);
            auto out1   = vector<double>(this->input);

            auto w2     = matrix<double>(this->hidden, this->output);
            auto delta2 = matrix<double>(this->hidden, this->output);
            auto in2    = vector<double>(this->hidden);
            auto out2   = vector<double>(this->hidden);
            auto theta2 = vector<double>(this->hidden);

            // Layer 3 - Output layer
            auto in3      = vector<double>(this->output);
            auto out3     = vector<double>(this->output);
            auto theta3   = vector<double>(this->output);
            auto expected = vector<double>(this->output);

            for (std::size_t i = 0; i < this->input; ++i) {
                for (std::size_t j = 0; j < this->hidden; ++j) {
                    w1[i][j] = double(std::rand() % 6) / 10.0;
                    if (std::rand() % 2) { w1[i][j] *= -1.0; }
                }
            }
            // Initialization for weights from Hidden layer to Output layer
            for (std::size_t i = 0; i < this->hidden; ++i) {
                for (std::size_t j = 0; j < this->output; ++j) {
                    w2[i][j] =
                        (double)(std::rand() % 10) / (10.0 * this->output);
                    if (std::rand() % 2) { w2[i][j] *= -1.0; }
                }
            }

            for (std::size_t __ei = 0; __ei < this->epochs; ++__ei) {
                std::cout << ">>> Beginning epoch " << __ei << std::endl;
#pragma omp for
                for (std::size_t sample = 0; sample < 50000ul; ++sample) {
                    for (std::size_t i = 0; i < 784; ++i) {
                        out1[i] = this->data.image[sample][i];
                    }
                    for (std::size_t i = 0; i < this->output; ++i) {
                        expected[i] = this->data.label[sample][i];
                    }
                    for (std::size_t i = 0; i < this->input; ++i) {
                        for (std::size_t j = 0; j < this->hidden; ++j) {
                            delta1[i][j] = 0.0;
                        }
                    }
                    for (std::size_t i = 0; i < this->hidden; ++i) {
                        for (std::size_t j = 0; j < this->output; ++j) {
                            delta2[i][j] = 0.0;
                        }
                    }

                    for (std::size_t i = 0; i < this->hidden; ++i) {
                        in2[i] = 0.0;
                    }
                    for (std::size_t i = 0; i < this->output; ++i) {
                        in3[i] = 0.0;
                    }
                    for (std::size_t i = 0; i < this->input; ++i) {
                        for (std::size_t j = 0; j < this->hidden; ++j) {
                            in2[j] += out1[i] * w1[i][j];
                        }
                    }
                    for (std::size_t i = 0; i < this->hidden; ++i) {
                        out2[i] = sigmoid(in2[i]);
                    }
                    for (std::size_t i = 0; i < this->hidden; ++i) {
                        for (std::size_t j = 0; j < this->output; ++j) {
                            in3[j] += out2[i] * w2[i][j];
                        }
                    }
                    for (std::size_t i = 0; i < this->output; ++i) {
                        out3[i] = sigmoid(in3[i]);
                    }

                    for (std::size_t i = 0; i < this->output; ++i) {
                        auto out3_sq = out3[i] * out3[i];
                        theta3[i]    = out3[i] * expected[i] - out3_sq -
                                    out3_sq * expected[i] + out3_sq * out3[i];
                    }
                    for (std::size_t i = 0; i < this->hidden; ++i) {
                        auto sum = std::inner_product(
                            w2[i].begin(), w2[i].end(), theta3.begin(), 0.0);
                        theta2[i] = sum * out2[i] - sum * out2[i] * out2[i];
                    }
                    for (std::size_t i = 0; i < this->hidden; ++i) {
                        for (std::size_t j = 0; j < this->output; ++j) {
                            delta2[i][j] =
                                this->__backprop(delta2, theta3, out2, i, j);
                            w2[i][j] += delta2[i][j];
                        }
                    }
                    for (std::size_t i = 0; i < this->input; ++i) {
                        for (std::size_t j = 0; j < this->hidden; j++) {
                            delta1[i][j] =
                                this->__backprop(delta1, theta2, out1, i, j);
                            w1[i][j] += delta1[i][j];
                        }
                    }

                    if ((sample + 1) % sample_per_thread == 0) {
                        for (std::size_t i = 0; i < this->input; ++i) {
                            for (std::size_t j = 0; j < this->hidden; j++) {
#pragma omp atomic
                                this->layers.at(0).weights[i][j] += w1[i][j];
                            }
                        }
                        for (std::size_t i = 0; i < this->hidden; ++i) {
                            for (std::size_t j = 0; j < this->output; j++) {
#pragma omp atomic
                                this->layers.at(1).weights[i][j] += w2[i][j];
                            }
                        }
                    }
                }
            }
        }
    }

    void write() {
        std::ofstream file("model-neural-network.dat", std::ios::out);

        // Input layer - Hidden layer
        for (std::size_t i = 0; i < this->input; ++i) {
            for (std::size_t j = 0; j < this->hidden; ++j) {
                file << this->layers.at(0).weights[i][j] << " ";
            }
            file << std::endl;
        }

        // Hidden layer - Output layer
        for (std::size_t i = 0; i < this->hidden; ++i) {
            for (std::size_t j = 0; j < this->output; ++j) {
                file << this->layers.at(1).weights[i][j] << " ";
            }
            file << std::endl;
        }

        file.close();
    }
};

// +--------------+
// | Main Program |
// +--------------+

void usage(const char* progname) {
    std::cout << "Usage: " << progname << " [options]" << std::endl;
    std::cout << "Program Options:\n" << std::endl;
    std::cout << "  -e  --epochs <VALUE>" << std::endl;
    std::cout << "  -l  --lrate <VALUE>" << std::endl;
    std::cout << "  -t  --threads <VALUE>" << std::endl;
    std::cout << "  -h  --help\n" << std::endl;
}

int main(int argc, char** argv) {
    std::size_t __count = 0ul;

    std::size_t threads = 0ul;
    std::size_t epochs  = 0ul;
    double      lrate   = 0ul;

    static struct option long_options[] = { { "help", 0, 0, 'h' },
                                            { "epochs", 1, 0, 'e' },
                                            { "lrate", 1, 0, 'l' },
                                            { "threads", 1, 0, 't' },
                                            { 0, 0, 0, 0 } };

    int opt;
    while ((opt = getopt_long(argc, argv, "e:l:t:h", long_options, NULL)) !=
           EOF) {
        switch (opt) {
            case 'e':
                __count++;
                std::sscanf(optarg, "%zu", &epochs);
                break;
            case 'l':
                __count++;
                std::sscanf(optarg, "%lf", &lrate);
                break;
            case 't':
                __count++;
                std::sscanf(optarg, "%zu", &threads);
                break;
            case 'h':
            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (__count < 3ul) {
        usage(argv[0]);
        return 1;
    }

    std::srand(std::time(nullptr));

    auto network = Network(threads, epochs, lrate);
    network.info();

    network.read_data();

    auto       clock      = std::chrono::system_clock();
    const auto time_begin = clock.now();

    network.train();

    const auto time_end = clock.now();

    const auto elapsed_secs =
        std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin);

    network.write();
    std::cout << "Elapsed time: " << elapsed_secs.count() << " seconds"
              << std::endl;

    return 0;
}