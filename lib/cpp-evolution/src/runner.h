#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

#include "config.h"
#include "population.h"
#include "random_wrapper.h"

#include "communicator.h"

// uncomment to compute and print a diversity metric in each iteration,
// expensive!
// #define PRINT_DIVERSITY 1

#ifndef PRINT_DIVERSITY
#define PRINT_DIVERSITY 0
#endif

template<typename elem_t, bool print_journal=false>
class Runner {
public:
    Runner(const Config &cfg,
            RandomWrapper &rw,
            typename elem_t::EvalInfo &eval_info,
            const std::vector<std::unique_ptr<elem_t>> &init_population,
            size_t num_mappings_to_print,
            bool print_as_json,
            std::ostream *journal_stream=nullptr,
            Communicator *communicator=nullptr)
        : Config_(cfg),
            RandomWrapper_(rw),
            EvalInfo_(eval_info),
            Population_(nullptr),
            PopSize_(cfg.getPopulationSize()),
            NumMappingsToPrint_(num_mappings_to_print),
            PrintAsJSON_(print_as_json),
            JournalStream_(journal_stream),
            Communicator_(communicator) {
        Population_ = Population<elem_t>::create(PopSize_, cfg.getMaxChildNum());

        if (Communicator_) {
            Communicator_->registerCommand({"print best",
                    [&](std::ostream& os, std::string_view) {
                        os << Population_->getPopAt(0) << std::endl;
                    }
                });
            Communicator_->registerCommand({"print all",
                    [&](std::ostream& os, std::string_view) {
                        for (auto it = Population_->getPopBegin(); it != Population_->getPopEnd(); ++it) {
                            os << **it << std::endl;
                        }
                    }
                });
        }

        for (auto &elem_ptr: init_population) {
            auto *entry = Population_->insertPop();
            entry->elem = *elem_ptr;
        }
        size_t already_inserted = init_population.size();

        for (size_t i = already_inserted; i < PopSize_; ++i) {
            auto *entry = Population_->insertPop();
            auto &elem = entry->elem;
            elem.initRandomly(RandomWrapper_, EvalInfo_);
        }

        Population_->finalize();

        evaluatePopulation();

        num_iterations = Config_.getNumIterations();
        num_restarts = Config_.getNumEpochs();
    }

    void doEvolution(void) {
        performEvolutionSteps();
        sortPopulation();
        std::cerr << "# Winning individuals:\n";
        for (size_t i = 0, e = std::min(PopSize_, NumMappingsToPrint_); i < e; i++) {
            if (PrintAsJSON_) {
                Population_->getPopAt(i).elem.dumpJson(std::cout);
            } else {
                std::cout << Population_->getPopAt(i);
            }
        }
    }

private:
    using it_t = typename Population<elem_t>::iterator_t;

    const Config &Config_;

    RandomWrapper &RandomWrapper_;

    typename elem_t::EvalInfo &EvalInfo_;

    typename std::unique_ptr<Population<elem_t>> Population_;

    const size_t PopSize_;
    const size_t NumMappingsToPrint_;

    size_t num_iterations;
    size_t num_restarts;

    const bool PrintAsJSON_;

    std::ostream *JournalStream_;

    Communicator *Communicator_;

    void jtime(void) {
        if (print_journal && JournalStream_ != nullptr) {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            *JournalStream_ << "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << "]";
        }
    }

    bool use_journal(void) const {
        return print_journal && JournalStream_ != nullptr;
    }

    void journal(const std::function<void(std::ostream&)>& stmt) {
        if (use_journal()) {
            stmt(*JournalStream_);
        }
    }

    void sortPopulation(void) {
        if (Config_.getEnableRatioCombination()) {
            Population_->ratio_sort();
        } else {
            Population_->rank_sort();
        }
    }

    void performEvolutionSteps(void) {
        size_t current_restart = 0;
        while (current_restart < num_restarts) {
            journal([&](auto &str){
                str << "starting epoch " << current_restart << " ";
                jtime();
                str << "\n";
            });
            if (current_restart != 0) {
                // re-randomize
                size_t keep = (Config_.getKeepRatio() * PopSize_) + 1;
                for (size_t i = keep; i < PopSize_; ++i) {
                    auto *entry = Population_->replacePop(i);
                    auto &elem = entry->elem;
                    elem.initRandomly(RandomWrapper_, EvalInfo_);
                }
                evaluatePopulation();
            }
            ++current_restart;

            sortPopulation();
            journal([&](auto &str){
                str << "  initial population:\n";
                const auto& best_val = (*Population_->getPopBegin())->fitness_val;
                const auto& worst_val = (*(Population_->getPopEnd()-1))->fitness_val;
                str << "    best:      " << best_val << "\n";
                str << "    worst:     " << worst_val << std::endl;
#if PRINT_DIVERSITY
                double div = Population_->computeDiversity();
                str << "    diversity: " << div << std::endl;
#endif
            });

            bool done = false;
            size_t current_it = 0;
            while (current_it < num_iterations && !done) {
                journal([&](auto &str){
                    str << "  generation " << current_it << " ";
                    jtime();
                    str << ":\n";
                });
                ++current_it;
                Population_->shuffle(RandomWrapper_);

                Population_->forall_chunks([&](auto start_it, auto end_it){
                    size_t num_recomb = Config_.getMaxRecombinationFactor() * (size_t)((double) ((end_it - start_it) / 2));
                    size_t num_mutations = Config_.getMaxMutationFactor() * (size_t)((double) ((end_it - start_it) / 2));
                    evolutionStep(start_it, end_it, num_recomb, num_mutations);
                });

                size_t curr_gen = Population_->getCurrentGeneration();
                sortPopulation();

                double luck_chance = Config_.getLuckChance();
                double bad_luck_protection = Config_.getBadLuckProtection();
                applyLuck(luck_chance, bad_luck_protection);

                Population_->purge();

                sortPopulation();

                double new_survivor_ratio = -1;
                double new_recomb_ratio = -1;
                double new_mut_ratio = -1;
                if (use_journal()) {
                    size_t num_new_survivors = 0;
                    size_t num_recomb_survivors = 0;
                    size_t num_mut_survivors = 0;
                    for (auto it = Population_->getPopBegin(); it != Population_->getPopEnd(); ++it) {
                        if ((*it)->birth_generation == curr_gen) {
                            num_new_survivors += 1;
                            if ((*it)->origin == Origin::Recombination) {
                                num_recomb_survivors += 1;
                            } else if ((*it)->origin == Origin::Mutation) {
                                num_mut_survivors += 1;
                            }
                        }
                    }
                    new_survivor_ratio = (double)num_new_survivors / (double)PopSize_;
                    new_recomb_ratio = (double)num_recomb_survivors / (double)PopSize_;
                    new_mut_ratio = (double)num_mut_survivors / (double)PopSize_;
                }

                const auto& best_val = (*Population_->getPopBegin())->fitness_val;
                const auto& worst_val = (*(Population_->getPopEnd()-1))->fitness_val;

                journal([&](auto &str){
                    str << "    best:      " << best_val << "\n";
                    str << "    worst:     " << worst_val << "\n";
#if PRINT_DIVERSITY
                    double div = Population_->computeDiversity();
                    str << "    diversity: " << div << "\n";
#endif
                    str << "    composition:\n";
                    str << "      old generation:   " << (1.0-new_survivor_ratio) * 100 << "%\n";
                    str << "      newly recombined: " << new_recomb_ratio * 100 << "%\n";
                    str << "      newly mutated:    " << new_mut_ratio * 100 << "%" << std::endl;
                    // Population_->printJournal(str, 2);
                });

                if (Communicator_) {
                    Communicator_->checkCommands();
                }

                if (best_val == worst_val) {
                    done = true;
                }
                if (best_val.is_optimal()) {
                    // We already found a perfect candidate, no need to go on
                    return;
                }
            }
            if (Config_.getEnableLocalOptimization()) {
                performLocalOptimization();
            }

            sortPopulation();
            const auto& best_val = (*Population_->getPopBegin())->fitness_val;
            const auto& worst_val = (*(Population_->getPopEnd()-1))->fitness_val;

            journal([&](auto &str){
                str << "    best:      " << best_val << "\n";
                str << "    worst:     " << worst_val << std::endl;
#if PRINT_DIVERSITY
                double div = Population_->computeDiversity();
                str << "    diversity: " << div << std::endl;
#endif
            });
        }
    }

    void performLocalOptimization(void) {
        journal([&](auto &str){
            str << "optimizing locally ";
            jtime();
            str << "\n";
        });
        Population_->forall_entries([&](auto &e){
            e.elem.optimizeLocally(e.fitness_val, EvalInfo_);
        });
    }


    /// Implementation of luck.
    ///
    /// For each to be purged individual, have it swap its place with a
    /// surviving individual at a probability of luck_chance. The best
    /// bad_luck_protection * population_size individuals cannot be the victim
    /// of such a swap.
    ///
    /// The intention is that this lowers selection pressure and therefore
    /// allows locally non-optimal but eventually beneficial structures to
    /// survive and reproduce.
    void applyLuck(double luck_chance, double bad_luck_protection) {
        size_t pop_end = Population_->getPopEndIdx();
        size_t children_end = Population_->getChildrenEndIdx();
        size_t first_unprotected = (size_t)(bad_luck_protection * pop_end);
        for (size_t i = pop_end; i < children_end; ++i) {
            if (RandomWrapper_.flip(luck_chance)) {
                size_t swapping_partner = RandomWrapper_.range(first_unprotected, pop_end - 1);
                Population_->swap(i, swapping_partner);
            }
        }
    }

    void evolutionStep(it_t pop_begin, it_t pop_end, size_t num_recomb, size_t num_mutations) {
        assert(pop_end - pop_begin > 0);

        auto &random = RandomWrapper_;

        for (size_t i = 0; i < num_mutations; ++i) {
            auto &parent = random.choice(pop_begin, pop_end)->elem;
            auto child = Population_->insertChild(Origin::Mutation);
            elem_t::mutate(random, child->elem, parent, EvalInfo_);
            child->evaluate(EvalInfo_);
        }

        for (size_t i = 0; i < num_recomb; ++i) {
            auto &parent_a = random.choice(pop_begin, pop_end)->elem;
            auto &parent_b = random.choice(pop_begin, pop_end)->elem;
            auto childA = Population_->insertChild(Origin::Recombination);
            auto childB = Population_->insertChild(Origin::Recombination);
            elem_t::recombine(random, childA->elem, childB->elem, parent_a, parent_b, EvalInfo_);
            childA->evaluate(EvalInfo_);
            childB->evaluate(EvalInfo_);
        }
    }

    void evaluatePopulation(void) {
        Population_->forall_entries([&](auto &e){
            e.evaluate(EvalInfo_);
        });
    }
};

