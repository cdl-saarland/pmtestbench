#include "mapping.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <unordered_set>

int Mapping::Fitness::compare(const Mapping::Fitness &a, const Mapping::Fitness &b, int group_idx) {
    if (a.is_infinity && b.is_infinity) {
        return 0;
    }
    if (a.is_infinity) {
        return 1;
    }
    if (b.is_infinity) {
        return -1;
    }

#define COMP(name, type, error, group)                                         \
    if (group == group_idx) {                                                  \
        if (std::abs((type)a.name - (type)b.name) > error) {                   \
            return (((type)a.name) < ((type)b.name) - error) ? -1 : 1;         \
        }                                                                      \
    }
#include "mapping_fitness.inc"
#undef COMP

    return 0; // a == b
}

float Mapping::Fitness::getComponentValue(int group_idx) const {
    if (this->is_infinity) {
        return INFINITY;
    }
    float res = 0.0;

#define COMP(name, type, error, group)                                         \
    if (group == group_idx) {                                                  \
        res += (float)this->name;                                              \
    }
#include "mapping_fitness.inc"
#undef COMP

    return res;
}

int Mapping::Fitness::getMaxGroup(void) {
    int res = 0;
#define COMP(name, type, error, group)                                         \
    res = std::max(res, group);
#include "mapping_fitness.inc"
#undef COMP
    return res;
}

bool Mapping::Fitness::operator==(const Mapping::Fitness &other) const {
    for (int i = 0; i <= getMaxGroup(); ++i) {
        if (0 != Mapping::Fitness::compare(*this, other, i)) {
            return false;
        }
    }
    return true;
}

bool Mapping::Fitness::is_optimal(void) const {
    return false;
    // A less pessimistic criterion might be interesting.
}

std::ostream& operator<< (std::ostream& stream, const Mapping::Fitness& fitness) {
    if (fitness.is_infinity) {
        stream << "infinity";
        return stream;
    }
    stream << "(";
#define COMP(name, type, error, group)                                         \
    stream << " ";                                                             \
    stream << #name "(";                                                       \
    if (group != COMP_DISABLED) {                                              \
        stream << "G" << group;                                                \
    } else {                                                                   \
        stream << "D";                                                         \
    }                                                                          \
    stream << "): " << fitness.get_##name();                                   \
    stream << " ";
#include "mapping_fitness.inc"
#undef COMP
    stream << ")";
    return stream;
}

void Mapping::EvalInfo::init_relevant_exps(void) {
    relevant_exps.resize(arch.getInstructions().size());
    for (const auto &e : exps) {
        for (const auto *i : e->getInsnSeq()) {
            relevant_exps.at(i->getID()).push_back(e.get());
        }
    }
}

Mapping::Mapping(void) {
}

bool Mapping::addEntry(const Instruction *insn, Uop uop, num_type num) {
    auto &vec_ptr = UopMap_[insn];
    if (vec_ptr.get() == nullptr) {
        vec_ptr.reset(new std::vector<std::tuple<Uop, num_type>>());
    }
    auto &vec = *vec_ptr;
    const auto& pos = std::lower_bound(vec.begin(), vec.end(), uop,
            [](auto a, auto b){return std::get<0>(a) < b;});
    if (pos != vec.end() && std::get<0>(*pos) == uop) {
        return false;
    }
    if (num > 0) {
        vec.insert(pos, std::tuple<Uop, num_type>(uop, num));
    }
    return true;
}

void Mapping::addInsn(const Instruction* insn) {
    auto &vec_ptr = UopMap_[insn];
    if (vec_ptr.get() == nullptr) {
        vec_ptr.reset(new std::vector<std::tuple<Uop, num_type>>());
    }
}

namespace{

#define NUM_LETTERS 26

void print_uop(std::ostream& stream, Uop uop) {
    assert((! (uop & ~((1 << NUM_LETTERS) - 1))) && "Too many ports in use!");

    for (unsigned i = 0; i < NUM_LETTERS; ++i) {
        if (uop & (1 << i)) {
            stream << (char)('A' + i);
        }
    }
}

void print_uop_json(std::ostream& stream, Uop uop) {
    stream << "[";
    bool first = true;
    for (unsigned i = 0; i < NUM_LETTERS; ++i) {
        if (uop & (1 << i)) {
            if (!first) {
                stream << ", ";
            }
            first = false;
            stream << i;
        }
    }
    stream << "]";
}

}

std::ostream& operator<< (std::ostream& stream, const Mapping& mapping) {
    mapping.dumpJson(stream);
    return stream;
}

void Mapping::dumpNonJson(std::ostream& stream) const {
    stream << "mapping: # {{{\n";
    for (const auto& [insn, vec] : this->UopMap_) {
        stream << "  " << *insn << ": # {{{\n";
        for (const auto& [uop, num] : *vec) {
            stream << "    ";
            print_uop(stream, uop);
            stream << ": " << num << "\n";
        }
        stream << "  # }}}\n\n";
    }
    stream << "# }}}";
}

void Mapping::dumpJson(std::ostream& stream) const {
    stream << "{\n";
    stream << "  \"kind\": \"Mapping3\",\n";
    stream << "  \"arch\": {\n";
    stream << "    \"kind\": \"Architecture\",\n";
    stream << "    \"insns\": [";
    bool first = true;

    for (const auto& [insn, vec] : this->UopMap_) {
        if (!first) {
            stream << ", ";
        }
        first = false;
        stream << "\"" << insn->getName() << "\"";
    }

    stream << "]\n";
    stream << "  },\n";
    stream << "\"assignment\": {\n";
    first = true;
    for (const auto& [I, vec] : this->UopMap_) {
        if (!first) {
            stream << ",\n";
        }
        first = false;
        stream << "    \"" << I->getName() << "\": [";
        bool inner_first = true;
        for (const auto& [uop, num] : *vec) {
            for (size_t j = 0; j < num; ++j) {
                if (!inner_first) {
                    stream << ", ";
                }
                inner_first = false;
                print_uop_json(stream, uop);
            }
        }
        stream << "]";
    }
    stream << "\n  }\n";
    stream << "}\n";
}

void Mapping::initRandomly(RandomWrapper &rw, const Mapping::EvalInfo &eval_info) {
    const auto &arch = eval_info.arch;

    std::vector<size_t> possible_indices;
    for (size_t i = 0; i < arch.getNumPorts(); ++i) {
        possible_indices.push_back(i);
    }
    std::vector<size_t> indices;

    for (const auto* insn : arch.getInstructions()) {
        float t = eval_info.getSingltonResult(insn);
        int num_distinct_uops = rw.range(1, arch.getNumPorts());
        // not formally connected, but a decent heuristic
        for (int  i = 0; i < num_distinct_uops; ++i) {
            int num_used_ports = rw.range(1, arch.getNumPorts());
            indices.clear();
            rw.sample(indices, possible_indices.begin(), possible_indices.end(), num_used_ports);
            Uop uop = 0;
            for (auto c : indices) {
                uop |= 1 << c;
            }
            int max_instances = (int)(t * (float) num_used_ports) + 1;
            int num_same_uops = rw.range(1, max_instances);
            this->addEntry(insn, uop, num_same_uops);
        }
    }
    this->normalize();
}

void Mapping::recombine(RandomWrapper &rw, Mapping &childA, Mapping &childB, const Mapping &parentA, const Mapping &parentB, const Mapping::EvalInfo &eval_info) {
    const auto &arch = eval_info.arch;

    for (const auto &insn : arch.getInstructions()) {
        std::vector<std::tuple<Uop, num_type>> uop_vec;
        for (auto &p : *parentA.UopMap_.at(insn)) {
            uop_vec.push_back(p);
        }
        for (auto &p : *parentB.UopMap_.at(insn)) {
            uop_vec.push_back(p);
        }
        rw.shuffle(uop_vec.begin(), uop_vec.end());
        size_t swap_point = rw.range(1, uop_vec.size() - 1);

        for (size_t i = 0; i < swap_point; ++i) {
            auto [u, n] = uop_vec[i];
            childA.addEntry(insn, u, n);
        }

        for (size_t i = swap_point; i < uop_vec.size(); ++i) {
            auto &[u, n] = uop_vec[i];
            childB.addEntry(insn, u, n);
        }
    }
    childA.normalize();
    childB.normalize();
}


namespace {

double rel_error(double estimate, double expectation, bool symmetric) {
    if (symmetric) {
        // weird symmetric relative error, custom definition
        // The interpretation is unclear, but it should punish over- and
        // under-estimations equally hard.
        return fabs(expectation - estimate) / std::min(expectation, estimate);
    } else {
        // relative error, usual definition
        // This substantially favors under-estimations.
        return fabs(expectation - estimate) / expectation;
    }
}

template<typename ELIST>
// This is necessary for the function to work for std::vector<Experiment*>
// as well as for std::vector<std::unique_ptr<Experiment>>
void evaluateImpl(const Mapping &m, Mapping::Fitness &res, const Architecture &arch, const Config &config, const ELIST &exps) {
    double max_diff = 0.0;
    double sum_diff = 0.0;
    double singleton_sum_diff = 0.0;
    int num_singletons = 0;

    double sum_ipc_diff = 0.0;

    double peak_ipc = config.getPeakIPC();
    double inverse_peak_ipc = -1.0;
    if (peak_ipc > 0.0) {
        inverse_peak_ipc = 1.0 / config.getPeakIPC();
    }

    for (const auto &e : exps) {
        double simulated_result = m.simulateExperiment(arch, *e);
        if (simulated_result == 0.0) {
            res = Mapping::Fitness{}.set_infinity();
            return;
        }
        if (peak_ipc > 0.0) {
            // This is how we handle frontend and retirement bottlenecks:
            // If the simulation result is faster than allowed by the
            // bottleneck, we assume that it executes at the limit set by the
            // bottleneck.
            size_t num_insns = e->getInsnSeq().size();
            double simulated_ipc = num_insns / simulated_result;
            if (simulated_ipc > peak_ipc) {
                simulated_result = num_insns * inverse_peak_ipc;
            }
        }
        double rel_diff = rel_error(simulated_result, e->getMeasuredCycles(), config.getEnableSymmetricError());

        if (rel_diff < 0.1) {
            // TODO check the effect of this
            rel_diff = 0.0;
        }
        max_diff = std::max(max_diff, rel_diff);
        sum_diff = sum_diff + rel_diff;

        double explen = e->getInsnSeq().size();
        double ref_ipc = (explen / e->getMeasuredCycles());
        double sim_ipc = (explen / simulated_result);

        sum_ipc_diff += rel_error(sim_ipc, ref_ipc, config.getEnableSymmetricError());

        if (e->getInsnSeq().size() == 1) {
            singleton_sum_diff += rel_diff;
            num_singletons += 1;
        }
    }
    double avg_diff = sum_diff / exps.size();
    double singleton_avg_diff = singleton_sum_diff / num_singletons;

    double avg_ipc_diff = sum_ipc_diff / exps.size();

    size_t num_uops = m.computeUopNumber();
    size_t uop_volume = m.computeUopVolume();

    float avg_num_diff_uops = m.computeAvgNumOfDifferentUops();

    res = Mapping::Fitness{};
    res.set_avg_err(avg_diff)
       .set_avg_ipc_err(avg_ipc_diff)
       .set_singleton_avg_err(singleton_avg_diff)
       .set_max_err(max_diff)
       .set_uop_volume(uop_volume)
       .set_uop_number(num_uops)
       .set_avg_num_diff_uops(avg_num_diff_uops);
}

}

void Mapping::evaluate(Mapping::Fitness &res, const Mapping::EvalInfo &eval_info) const {
    const auto &arch = eval_info.arch;
    const auto &config = eval_info.config;
    const auto &exps = eval_info.exps;
    evaluateImpl(*this, res, arch, config, exps);
}

void Mapping::evaluateInsn(Mapping::Fitness &res, const Mapping::EvalInfo &eval_info, const Instruction* insn) const {
    const auto &arch = eval_info.arch;
    const auto &config = eval_info.config;
    const auto &exps = eval_info.getRelevantExps(insn);
    evaluateImpl(*this, res, arch, config, exps);
}


void Mapping::optimizeLocally(Mapping::Fitness &res, const Mapping::EvalInfo &eval_info) {
    const auto &arch = eval_info.arch;

    Mapping::Fitness prev_fitness;
    Mapping::Fitness new_fitness;

    for (const auto *i : arch.getInstructions()) {
        evaluateInsn(prev_fitness, eval_info, i);

        auto prev_vec = UopMap_[i];
        // Create a copy of the vector for instruction i
        auto new_vec = std::make_shared<std::vector<std::tuple<Uop, num_type>>>(*prev_vec);

        // Set the corresponding entry for instruction i to the new vector
        UopMap_[i] = new_vec;

        int max_idx = UopMap_[i]->size();

        bool changed = false;

        for (int idx = 0; idx < max_idx; ++idx) {
            auto& tup = (*new_vec)[idx];

            num_type n_before = std::get<1>(tup);
            auto& n_ref = std::get<1>(tup);

            if (n_ref == 0) {
                continue;
            }

            // see whether results get better if we reduce the uop number
            n_ref -= 1;
            evaluateInsn(new_fitness, eval_info, i);
            if (Fitness::compare(new_fitness, prev_fitness) <= 0) {
                // reducing improves things, so see how much to reduce
                changed = true;
                prev_fitness = new_fitness;
                while (n_ref > 0) {
                    n_ref -= 1;
                    evaluateInsn(new_fitness, eval_info, i);
                    if (! (Fitness::compare(new_fitness, prev_fitness) <= 0)) {
                        // we reduced to much
                        n_ref += 1;
                        break;
                    }
                    prev_fitness = new_fitness;
                }
                continue;
            }

            n_ref = n_before;

            // see whether results get better if we increase the uop number
            n_ref += 1;
            evaluateInsn(new_fitness, eval_info, i);
            if (Fitness::compare(new_fitness, prev_fitness) < 0) {
                // increasing improves things, so see how much to increase
                changed = true;
                prev_fitness = new_fitness;
                while (true) {
                    n_ref += 1;
                    evaluateInsn(new_fitness, eval_info, i);
                    if (! (Fitness::compare(new_fitness, prev_fitness) < 0)) {
                        // we increased to much
                        n_ref -= 1;
                        break;
                    }
                    prev_fitness = new_fitness;
                }
                continue;
            }

            n_ref = n_before;
        }
        if (! changed) {
            // if nothing changed, avoid the unnecessary newly allocated vector
            UopMap_[i] = prev_vec;
        }
    }

    // normalize, especially remove any n == 0 uops
    this->normalize();

    evaluate(res, eval_info);
}

void Mapping::normalize(void) {
    for (const auto &[insn, uops]: this->UopMap_) {
        std::sort(uops->begin(), uops->end());
        auto sz = uops->size();
        // for each distinct uop, accumulate their co-efficient in the last
        // corresponding entry, setting all previous ones to 0
        for (size_t i = 1; i < sz; ++i) {
            auto &[u_a, n_a] = (*uops)[i - 1];
            auto &[u_b, n_b] = (*uops)[i];
            if (u_a == u_b) {
                n_b += n_a;
                n_a = 0;
            }
        }
        // remove n==0 entries
        uops->erase(
                std::remove_if(uops->begin(), uops->end(),
                    [](const auto &x){ return std::get<1>(x) == 0; }),
                uops->end()
            );
    }
}

double Mapping::distance(const Mapping &a, const Mapping &b) {
    double result = 0;

    for (const auto &[insn, uops_a] : a.UopMap_) {
        double insn_result = 0;
        const auto &uops_b = b.UopMap_.at(insn);
        // compute the number of uops that occur in one mapping but not the
        // other, divide by the sum of the numbers of uops in both

        size_t a_sz = uops_a->size();
        size_t b_sz = uops_b->size();

        size_t a_idx = 0;
        size_t b_idx = 0;

        size_t total_uops = 0;

        while ((a_idx < a_sz) && (b_idx < b_sz)) {
            const auto &[u_a, n_a] = uops_a->at(a_idx);
            const auto &[u_b, n_b] = uops_b->at(b_idx);
            if (u_a < u_b) {
                insn_result += n_a;
                total_uops += n_a;
                ++a_idx;
            } else if (u_b < u_a) {
                insn_result += n_b;
                total_uops += n_b;
                ++b_idx;
            } else {
                insn_result += std::abs((int)n_a - (int)n_b);
                total_uops += n_a + n_b;
                ++a_idx;
                ++b_idx;
            }
        }
        while (a_idx < a_sz) {
            const auto &[u_a, n_a] = uops_a->at(a_idx);
            insn_result += n_a;
            total_uops += n_a;
            ++a_idx;
        }
        while (b_idx < b_sz) {
            const auto &[u_b, n_b] = uops_b->at(b_idx);
            insn_result += n_b;
            total_uops += n_b;
            ++b_idx;
        }
        result += insn_result / (double)total_uops;
    }

    return result;
}

size_t Mapping::computeUopNumber(void) const {
    std::vector<Uop> used_uops;
    for (const auto &[i, uops] : UopMap_) {
        for (const auto &[u, n] : *uops) {
            used_uops.push_back(u);
        }
    }
    std::sort(used_uops.begin(), used_uops.end());

    size_t num = 0;
    Uop prev_uop = (Uop)-1;
    for (auto u : used_uops) {
        num += (prev_uop != u);
        prev_uop = u;
    }

    return num;
}

size_t Mapping::computeUopVolume(void) const {
    size_t res = 0;
    for (const auto &[i, uops] : UopMap_) {
        for (const auto &[u, n] : *uops) {
            res += __builtin_popcount(u) * n;
        }
    }
    return res;
}

float Mapping::computeAvgNumOfDifferentUops(void) const {
    size_t num_differing_uops = 0;
    size_t num_insns = 0;

    for (const auto &[i, uops] : UopMap_) {
        num_differing_uops += uops->size();
        num_insns += 1;
    }
    return (float)num_differing_uops / (float)num_insns;
}

void Mapping::mutate(RandomWrapper &rw, Mapping &child, const Mapping &parent, const EvalInfo &eval_info) {
    const auto &arch = eval_info.arch;
    const auto &cfg = eval_info.config;

    double add_uop = cfg.getMutAddUopChance();
    double change_uop = cfg.getMutChangeUopChance();
    double change_n = cfg.getMutChangeNumChance();

    for (const auto &insn : arch.getInstructions()) {
        std::vector<std::tuple<Uop, num_type>> uop_vec;
        const auto& parent_vec = *parent.UopMap_.at(insn);
        num_type total_n = 0;
        for (auto &[u, n] : parent_vec) {
            total_n += n;
        }
        for (auto &[u, n] : parent_vec) {
            Uop this_u = u;
            num_type this_n = n;
            if (rw.flip(change_uop)) {
                const auto *other_insn = rw.choice(arch.getInstructions());
                const auto &other_insn_vec = *parent.UopMap_.at(other_insn);
                const auto &[new_u, other_n] = rw.choice(other_insn_vec);
                this_n = (this_n * __builtin_popcount(new_u)) / __builtin_popcount(this_u);
                this_u = new_u;
            } else if (rw.flip(change_n)) {
                if (rw.flip(0.5)) {
                    this_n += 1;
                    total_n += 1;
                } else if (total_n > 1) {
                    this_n -= 1;
                    total_n -= 1;
                }
            }
            if (this_n <= 0) {
                this_n = 1;
            }
            child.addEntry(insn, this_u, this_n);
        }
        while (rw.flip(add_uop)) {
            const auto *other_insn = rw.choice(arch.getInstructions());
            const auto &other_insn_vec = *parent.UopMap_.at(other_insn);
            const auto &[new_u, other_n] = rw.choice(other_insn_vec);
            child.addEntry(insn, new_u, rw.range(1, other_n));
        }
    }

    child.normalize();
}
