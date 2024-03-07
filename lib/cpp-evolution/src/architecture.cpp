#include "architecture.h"

using namespace std;

const std::vector<Instruction*> &Architecture::getInstructions(void) const {
    return Instructions_;
}

Architecture::Architecture(size_t num_ports) {
    setNumPorts(num_ports);
}

Instruction *Architecture::getInstruction(std::string_view name) {
    Instruction *res = nullptr;
    const auto &insn = NameMap_.find(std::string(name));
    if (insn == NameMap_.end()) {
        res = new Instruction(name);
        NameMap_[res->getName()] = res; // avoid out of lifetime references
        Instructions_.push_back(res);
    } else {
        res = insn->second;
    }
    return res;
}

void Architecture::setNumPorts(size_t n) {
    NumPorts_ = n;
}

size_t Architecture::getNumPorts(void) const {
    return NumPorts_;
}

Uop Architecture::getLargestUop(void) const {
    return (1 << getNumPorts()) - 1;
}

Architecture::~Architecture(void) {
    for (auto& insn : Instructions_) {
        delete insn;
    }
}
