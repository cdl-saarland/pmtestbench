#include "experiment.h"

#include <assert.h>

Experiment::Experiment(const std::vector<Instruction*> &insnSeq) : InsnSeq_(insnSeq) {
}

const std::vector<Instruction*> &Experiment::getInsnSeq(void) const {
    return InsnSeq_;
}

double Experiment::getMeasuredCycles(void) const {
    assert(isEvaluated);
    return MeasuredCycles_;
}

void Experiment::setMeasuredCycles(double cycles) {
    isEvaluated = true;
    MeasuredCycles_ = cycles;
}

// Experiment *Experiment::fromRandom(seed_type seed, const Config& cfg) {
//     std::vector<Instruction*> insnSeq;
//     // TODO
//     return new Experiment(insnSeq);
// }
//
std::ostream& operator<< (std::ostream& stream, const Experiment& e) {
    stream << "experiment:\n";
    stream << "  instructions:\n";
    for (auto insn : e.InsnSeq_) {
        stream << "    " << *insn << "\n";
    }
    stream << "  cycles: ";
    if (e.isEvaluated) {
        stream << "    " << e.MeasuredCycles_ << "\n";
    } else {
        stream << "    none\n";
    }

    return stream;
}
