#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "config.h"
#include "instruction.h"

class Experiment {
public:
    Experiment(const std::vector<Instruction*> &insnSeq);

    const std::vector<Instruction*> &getInsnSeq(void) const;

    double getMeasuredCycles(void) const;

    void setMeasuredCycles(double cycles);

    // static Experiment *fromRandom(seed_type seed, const Config& cfg);

    friend std::ostream& operator<< (std::ostream& stream, const Experiment& e);

private:
    bool isEvaluated = false;

    std::vector<Instruction*> InsnSeq_;

    double MeasuredCycles_;
};

using ExpVec = std::vector<std::unique_ptr<Experiment>>;

