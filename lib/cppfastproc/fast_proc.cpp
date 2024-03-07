
#include <cstdint>
#include <vector>
#include <iostream>

class FasterProcessor {
public:
    FasterProcessor(uint32_t n);

    void add(uint32_t uop, uint32_t n);

    void clear(void);

    double compute(void);

private:
    uint32_t numPorts;
    std::vector<uint32_t> uops;
    std::vector<uint32_t> numbers;
};

FasterProcessor::FasterProcessor(uint32_t n) {
    this->numPorts = n;
}

void FasterProcessor::add(uint32_t uop, uint32_t n) {
    this->uops.push_back(uop);
    this->numbers.push_back(n);
}

void FasterProcessor::clear(void) {
    this->uops.clear();
    this->numbers.clear();
}

double FasterProcessor::compute(void) {
    double max_val = 0.0;
    uint32_t max_uop = 1 << this->numPorts;

    auto& uops = this->uops;
    auto& numbers = this->numbers;
    size_t max_i = uops.size();
    for (uint32_t current_q = 1; current_q < max_uop; ++current_q) {
        double val = 0.0;
        for (size_t i = 0; i < max_i; ++i) {
            if ((~current_q & uops[i]) == 0){
                val += numbers[i];
            }
        }
        val = val / __builtin_popcount(current_q);
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val;
}

#ifndef NOPYBIND
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cppfastproc, m) {
    py::class_<FasterProcessor>(m, "FP")
        .def(py::init<int>())
        .def("add", &FasterProcessor::add)
        .def("clear", &FasterProcessor::clear)
        .def("compute", &FasterProcessor::compute);
}

#endif


int main(void) {
    auto fp = FasterProcessor(3);
    fp.add(04, 1); // mul
    fp.add(06, 2); // add
    fp.add(01, 1); // store

    auto res = fp.compute();

    std::cout << "Result:" << res << "\n";
}
