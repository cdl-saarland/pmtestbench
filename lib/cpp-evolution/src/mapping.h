#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "architecture.h"
#include "config.h"
#include "experiment.h"
#include "instruction.h"
#include "population.h"
#include "random_wrapper.h"


using num_type = uint32_t;

/// mapping from instruction to uop and number
class Mapping {
public:
    class Fitness {
    public:
        Fitness(void) {}

        // define component getters
#define COMP(name, type, error, group) \
        type get_##name(void) const { return name; }
#include "mapping_fitness.inc"
#undef COMP

        // define component setters
#define COMP(name, type, error, group) \
        Fitness& set_##name(type val) { this->name = val; return *this; }
#include "mapping_fitness.inc"
#undef COMP

        float getComponentValue(int group_idx) const;

        Fitness& set_infinity(void) {
            is_infinity = true;
            return *this;
        }

        bool operator==(const Fitness&) const;

        bool is_optimal(void) const;

        friend std::ostream& operator<< (std::ostream& stream, const Fitness& fitness);

        /// Comparator function, returns -1 if a < b, 0 if a == b
        /// and 1 if a > b according to the group of fitness components
        /// indicated by group_idx.
        static int compare(const Fitness &a, const Fitness &b, int group_idx=0);

        /// Get the maximal group_idx
        static int getMaxGroup(void);

    private:
        // declare fields for all components defined in the config file
#define COMP(name, type, error, group) \
        type name;
#include "mapping_fitness.inc"
#undef COMP

        bool is_infinity = false;
    };

    class EvalInfo {
    public:
        EvalInfo(const Architecture &arch, const ExpVec &exps, const std::vector<float> &singleton_results, const Config &config):
            arch(arch), exps(exps), config(config), singleton_results(singleton_results) {
                init_relevant_exps();
            }

        const Architecture &arch;
        const ExpVec &exps;
        const Config &config;

        float getSingltonResult(const Instruction* insn) const {
            return singleton_results.at(insn->getID());
        }

        const std::vector<const Experiment*> &getRelevantExps(const Instruction *i) const {
            return relevant_exps.at(i->getID());
        }

    private:
        const std::vector<float> &singleton_results;
        void init_relevant_exps(void);
        std::vector<std::vector<const Experiment*>> relevant_exps;
    };

    Mapping(void);

    bool addEntry(const Instruction* insn, Uop uop, num_type num);

    void addInsn(const Instruction* insn);

    friend std::ostream& operator<< (std::ostream& stream, const Mapping& mapping);

    /// Initialize the mapping randomly.
    /// The resulting mapping is normalized.
    void initRandomly(RandomWrapper &rw, const EvalInfo &eval_info);

    /// Fill the childrem mappings with information from the parents.
    /// The resulting children are normalized.
    static void recombine(RandomWrapper &rw, Mapping &childA, Mapping &childB, const Mapping &parentA, const Mapping &parentB, const EvalInfo &eval_info);

    static void mutate(RandomWrapper &rw, Mapping &child, const Mapping &parent, const EvalInfo &eval_info);

    double simulateExperiment(const Architecture &arch, const Experiment &e) const;

    void evaluate(Fitness &res, const EvalInfo &eval_info) const;

    void dumpJson(std::ostream& stream) const;

    void dumpNonJson(std::ostream& stream) const;

    /// Try small local improvements in the style of hill climbing to improve
    /// the fitness of the mapping. The resulting fitness is written into res.
    /// The resulting mapping is normalized.
    void optimizeLocally(Fitness &res, const EvalInfo &eval_info);

    /// For each instruction, sort the uop vector, merge entries for the same
    /// uops and remove uop entries with co-efficient 0.
    void normalize(void);

    /// Compute a metric for the distance between two normalized mappings.
    static double distance(const Mapping &a, const Mapping &b);

    size_t computeUopNumber(void) const;

    size_t computeUopVolume(void) const;

    float computeAvgNumOfDifferentUops(void) const;

private:
    std::map<const Instruction*, std::shared_ptr<std::vector<std::tuple<Uop, num_type>>>> UopMap_;

    void evaluateInsn(Fitness &res, const EvalInfo &eval_info, const Instruction* insn) const;
};
