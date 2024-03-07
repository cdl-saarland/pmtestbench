#include "random_wrapper.h"

#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif


RandomWrapper::RandomWrapper(RandomWrapper::seed_t s) {
#ifdef _OPENMP
    int n = omp_get_max_threads();
    for (int i = 0; i < n; ++i) {
        RNGs_.emplace_back();
    }
#else
    RNGs_.emplace_back();
#endif
    this->seed(s);
}

void RandomWrapper::seed(RandomWrapper::seed_t s) {
    int i = 0;
    for (auto &rng : RNGs_) {
        rng.seed(s + i);
        ++i;
    }
}

RandomWrapper::generator_t &RandomWrapper::getCurrentRNG(void) {
#ifdef _OPENMP
    return RNGs_[omp_get_thread_num()];
#else
    return RNGs_[0];
#endif
}

int64_t RandomWrapper::range(int64_t base, int64_t bound) {
    assert(base <= bound);
    std::uniform_int_distribution<int64_t> uniform_dist(base, bound);
    return uniform_dist(getCurrentRNG());
}

int64_t RandomWrapper::range(int64_t bound) {
    return this->range(0, bound);
}

bool RandomWrapper::flip(double true_chance) {
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    return uniform_dist(getCurrentRNG()) <= true_chance;
}
