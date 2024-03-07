#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "random_wrapper.h"

enum class Origin {
    Initialization = 0,
    Recombination = 1,
    Mutation = 2,
};

static inline std::ostream& operator<< (std::ostream& stream, const Origin& o) {
    switch (o) {
        case Origin::Initialization:
            stream << "Initialization";
            break;
        case Origin::Recombination:
            stream << "Recombination";
            break;
        case Origin::Mutation:
            stream << "Mutation";
            break;
    }
    return stream;
}

/** A data structure for representing the population in a genetic algorithm
 *
 * All functionality except non-modifying iteration and insertion of new
 * children is not thread safe.
 *
 * The template parameter `elem_t` is the type of the individuals in the
 * population. The following member types have to be defined:
 *    - `elem_t::Fitness` for a representation of its fitness
 *    - `elem_t::EvalInfo` for data that are required for evaluation
 */
template<typename elem_t>
class Population {
public:
    struct Entry {
        elem_t elem;
        typename elem_t::Fitness fitness_val;
        size_t accumulated_position = 0; // used for rank-combined sorting
        float accumulated_value = 0.0; // used for ratio-combined sorting
        bool evaluated = false;
        size_t birth_generation;
        Origin origin;

        Entry(size_t birth_generation, Origin origin) : birth_generation(birth_generation), origin(origin) { }

        void evaluate(const typename elem_t::EvalInfo &eval_info) {
            if (evaluated) {
                return;
            }
            elem.evaluate(this->fitness_val, eval_info);
            evaluated = true;
        }

        // friend bool operator< ( Entry const& a, Entry const& b) {
        //     return a.fitness_val < b.fitness_val;
        // }

        friend std::ostream& operator<< (std::ostream& stream, const Entry& entry) {
            stream << entry.elem << "\n# with fitness value ";
            stream << entry.fitness_val;
            stream << "\n# created in generation " << entry.birth_generation << " from " << entry.origin;
            stream << "\n";
            return stream;
        }
    };

    using iterator_t = typename std::vector<Entry*>::iterator;

    static typename std::unique_ptr<Population<elem_t>> create(size_t pop_size, size_t child_num) {
        return std::unique_ptr<Population>(new Population(pop_size, child_num));
    }

    ~Population(void) {
        // creepy things we have to do because of placement new
        for (size_t i = 0; i < first_free_pop; ++i) {
            Arena[i]->~Entry();
        }
        for (size_t i = start_dead_zone; i < first_free_child; ++i) {
            Arena[i]->~Entry();
        }
        free(Storage);
    }

    void shuffle(RandomWrapper &rw) {
        assert(finalized);
        rw.shuffle(Arena.begin(), Arena.begin() + first_free_pop);
    }

    /// Sort the entire population (children and residual individuals).
    /// Sorting works as follows:
    /// The fitness type is expected to define a set of groups of fitness
    /// orderings (based on serveral different fitness aspects).
    /// The population is sorted wrt. each of these fitness ordering groups and
    /// for each sorting, the respective position is annotated to the
    /// individuals. The final sorting is done wrt. the sums of positions.
    /// Effectively, individuals are sorted wrt. the arithmetic mean of their
    /// performance in all ordering groups.
    void rank_sort(void) {
        assert(finalized);
        assert(first_free_pop == start_dead_zone);

        auto end_it = Arena.begin() + first_free_child;

        // initialize the accumulated sum of all positions for each individual
        for (auto it = Arena.begin(); it != end_it; ++it) {
            (*it)->accumulated_position = 0;
        }

        for (int group_idx = 0; group_idx <= elem_t::Fitness::getMaxGroup(); ++group_idx) {
            // sort wrt. each ordering group...
            std::sort(Arena.begin(), end_it,
                    [&](const auto& a, const auto &b){
                        return elem_t::Fitness::compare(a->fitness_val, b->fitness_val, group_idx) == -1;
                    }
                );
            // ...and accumulate the respective positions for each individual
            size_t idx = 0;
            for (auto it = Arena.begin(); it != end_it; ++it) {
                (*it)->accumulated_position += idx;
                ++idx;
            }
        }

        // final sorting according to the accumulated positions
        std::sort(Arena.begin(), end_it,
                [](const auto& a, const auto &b){
                    return a->accumulated_position < b->accumulated_position;
                }
            );
    }

    /// Sort the entire population like rank_sort, however do not use the
    /// positions after sorting but use for each individual the sum of all
    /// component fitness values after applying a linear scale to a fixed
    /// interval.
    /// This puts more focus the magnitude of fitness improvements.
    void ratio_sort(void) {
        assert(finalized);
        assert(first_free_pop == start_dead_zone);

        auto end_it = Arena.begin() + first_free_child;

        // initialize the accumulated sum of all positions for each individual
        for (auto it = Arena.begin(); it != end_it; ++it) {
            (*it)->accumulated_value = 0.0;
        }

        float range_min = 1;
        float range_max = 1000;

        for (int group_idx = 0; group_idx <= elem_t::Fitness::getMaxGroup(); ++group_idx) {
            float max_val = 0.0;
            float min_val = 0.0;
            for (auto it = Arena.begin(); it != end_it; ++it) {
                float val = (*it)->fitness_val.getComponentValue(group_idx);
                max_val = std::max(max_val, val);
                min_val = std::min(min_val, val);
            }
            for (auto it = Arena.begin(); it != end_it; ++it) {
                float val = (*it)->fitness_val.getComponentValue(group_idx);

                float x;
                // apply a linear transform to map it into [range_min, range_max]
                if (max_val == min_val) {
                    x = range_min;
                } else {
                    x = (((range_max - range_min) * (val - min_val)) / (max_val - min_val)) + 1;
                }

                (*it)->accumulated_value += x;
            }
        }

        // final sorting according to the accumulated positions
        std::sort(Arena.begin(), end_it,
                [](const auto& a, const auto &b){
                    return a->accumulated_value < b->accumulated_value;
                }
            );
    }

    void swap(size_t idx1, size_t idx2) {
        assert(finalized);
        assert(0 <= idx1 && idx1 < first_free_child);
        assert(0 <= idx2 && idx2 < first_free_child);
        auto tmp = Arena[idx1];
        Arena[idx1] = Arena[idx2];
        Arena[idx2] = tmp;
    }

    size_t getPopEndIdx(void) {
        return first_free_pop;
    }

    size_t getChildrenEndIdx(void) {
        return first_free_child;
    }

    void purge(void) {
        assert(finalized);
        // creepy things we have to do because of placement new
        for (size_t i = start_dead_zone; i < first_free_child; ++i) {
            Arena[i]->~Entry();
        }
        first_free_child = start_dead_zone;
        current_generation += 1;
    }

    Entry *insertPop(Origin origin=Origin::Initialization) {
        assert(not finalized);

        size_t pos = first_free_pop.fetch_add(1);

        assert(pos < start_dead_zone);
        Entry *res = Arena[pos];

        //placement new
        new (res) Entry(current_generation, origin);

        return res;
    }

    Entry *replacePop(size_t idx, Origin origin=Origin::Initialization) {
        auto* e = Arena[idx];
        e->~Entry();
        new (e) Entry(current_generation, origin);
        return e;
    }

    Entry *insertChild(Origin origin) {
        assert(finalized);

        size_t pos = first_free_child.fetch_add(1);

        assert(pos < num_elements);

        Entry *res = Arena[pos];

        //placement new
        new (res) Entry(current_generation, origin);

        return res;
    }

    typename std::vector<Entry*>::iterator getPopBegin(void) {
        assert(finalized);
        return Arena.begin();
    }

    typename std::vector<Entry*>::const_iterator getPopBegin(void) const {
        assert(finalized);
        return Arena.begin();
    }

    typename std::vector<Entry*>::iterator getPopEnd(void) {
        assert(finalized);
        return Arena.begin() + first_free_pop;
    }

    typename std::vector<Entry*>::const_iterator getPopEnd(void) const {
        assert(finalized);
        return Arena.begin() + first_free_pop;
    }

    void forall_entries(std::function<void(Entry&)> stmt) {
        size_t pop_size = first_free_pop;
        #pragma omp parallel for
        for (size_t i = 0;  i < pop_size; ++i) {
            auto &entry = this->getPopAt(i);
            stmt(entry);
        }
    }

    void forall_chunks(std::function<void(typename std::vector<Entry*>::iterator, typename std::vector<Entry*>::iterator)> stmt) {
        auto pop_begin = this->getPopBegin();
        #pragma omp parallel for
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t chunk_start = i * chunk_size;
            size_t chunk_end = std::min(chunk_start + chunk_size, (size_t)first_free_pop);
            auto start_it = pop_begin + chunk_start;
            auto end_it = pop_begin + chunk_end;

            stmt(start_it, end_it);
        }
    }

    Entry& getPopAt(size_t idx) {
        assert(finalized);
        assert((0 <= idx) && (idx < first_free_pop));
        return *Arena.at(idx);
    }

    /** Finalize construction phase of the initial population
     *
     * This checks several invariants that have to hold for the population at
     * any point after calling this function. This method has to be called
     * before new children can be added or iterated.
     */
    void finalize(void) {
        assert(num_elements > start_dead_zone);
        assert(first_free_pop == start_dead_zone);
        assert(first_free_child == start_dead_zone);
        assert(start_dead_zone > 0);

        current_generation += 1;

        finalized = true;
    }

    void printJournal(std::ostream& journal, int indent) {
        add_indent(journal, indent);
        journal << "[\n";
        bool first = true;
        for (auto it = this->getPopBegin(); it != this->getPopEnd(); ++it) {
            if (!first) {
                journal << ",\n";
            }
            first = true;
            add_indent(journal, indent + 2);
            journal << "\"";
            journal << (*it)->fitness_val;
            journal << "\"";
        }
        journal << "\n";
        add_indent(journal, indent);
        journal << "]\n";
    }

    double computeDiversity(void) const {
        double result = 0;

        auto pop_end = getPopEnd();
        for (auto it_a = getPopBegin(); it_a < pop_end; ++it_a) {
            for (auto it_b = it_a + 1; it_b < pop_end; ++it_b) {
                result += elem_t::distance((*it_a)->elem, (*it_b)->elem);
            }
        }

        return result / (double)first_free_pop;
    }

    size_t getCurrentGeneration(void) {
        return current_generation;
    }

private:
    Population(size_t pop_size, size_t child_num) :
            num_elements(pop_size + child_num), Arena(num_elements) {
        start_dead_zone = pop_size;

        Storage = (Entry*)malloc(num_elements * sizeof(Entry));

        for (size_t i = 0; i < num_elements; ++i) {
            Arena[i] = &Storage[i];
        }

        first_free_pop = 0;
        first_free_child = start_dead_zone;

#ifdef _OPENMP
        num_chunks = omp_get_max_threads();
#else
        num_chunks = 1;
#endif
        chunk_size = pop_size / num_chunks;
    }

    void add_indent(std::ostream& stream, int indent) {
        for (int i = 0; i < indent; ++i) {
            stream << " ";
        }
    }

    size_t num_elements;
    size_t start_dead_zone;
    std::atomic<size_t> first_free_pop;
    std::atomic<size_t> first_free_child;

    size_t num_chunks = 1;
    size_t chunk_size;


    bool finalized = false;

    size_t current_generation = 0;

    std::vector<Entry*> Arena;

    Entry *Storage;

};


