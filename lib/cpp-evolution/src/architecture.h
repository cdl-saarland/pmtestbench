#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "instruction.h"

using Uop = uint_fast32_t;

class Architecture {
public:
    const std::vector<Instruction*> &getInstructions(void) const;

    Instruction *getInstruction(std::string_view name);

    size_t getNumPorts(void) const;

    void setNumPorts(size_t n);

    Uop getLargestUop(void) const;

    ~Architecture(void);

    Architecture(size_t num_ports=8);

    // TODO make only constructable via static factory function with unique_ptr

private:
    std::vector<Instruction*> Instructions_;
    std::map<std::string, Instruction*> NameMap_;
    size_t NumPorts_ = 0;
};

