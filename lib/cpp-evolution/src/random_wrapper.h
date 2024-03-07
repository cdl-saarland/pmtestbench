#pragma once

#include <algorithm>
#include <random>
#include <vector>
#include <cstdint>
#include <iostream>

/**
 *  Wrapper class to encapsulate the ugly C++11 random interface.
 *  Querrying works thread-safely with OpenMP with multiple generators.
 *
 */
class RandomWrapper {
public:
    // this defines which random number generator to use
    using generator_t = std::mt19937_64;
    using seed_t = uint32_t;

    RandomWrapper(RandomWrapper::seed_t s=0);

    /** Seed the random number generator
     */
    void seed(RandomWrapper::seed_t s);

    /** Produce a random number in the closed interval [base, bound].
     */
    int64_t range(int64_t base, int64_t bound);

    /** Produce a random number in the closed interval [0, bound].
     */
    int64_t range(int64_t bound);

    /** Randomly shuffle the part of a collection between the iterators begin
     * (inclusively) and end (exclusively).
     */
    template<typename it_t>
    void shuffle(it_t begin, it_t end) {
        std::shuffle(begin, end, getCurrentRNG());
    }

    /** Return true or false, the probability for true is true_chance.
     */
    bool flip(double true_chance=0.5);

    /** Randomly select an element of a collection between the iterators begin
     * (inclusively) and end (exclusively).
     */
    template<typename it_t>
    typename std::iterator_traits<it_t>::value_type choice(const it_t begin, const it_t end) {
        auto num_elements = end - begin;
        auto idx = this->range(0, num_elements-1);
        return *(begin + idx);
    }

    /** Randomly select an element of a vector.
     */
    template<typename elem_t>
    const elem_t choice(const typename std::vector<elem_t> &vec) {
        return choice(vec.begin(), vec.end());
    }

    /** Randomly sample multiple element of a collection between the iterators
     * begin (inclusively) and end (exclusively) without redundant objects.
     */
    template<typename it_t>
    void sample(std::vector<typename std::iterator_traits<it_t>::value_type> &dest, it_t begin, it_t end, size_t num) {
        std::sample(begin, end, std::back_inserter(dest), num, getCurrentRNG());
    }

    void sample_indices(std::vector<size_t> &dest, size_t num, size_t max) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < max; ++i) {
            indices.push_back(i);
        }
        this->sample(dest, indices.begin(), indices.end(), num);
    }

private:
    std::vector<generator_t> RNGs_;
    generator_t &getCurrentRNG(void);
};

