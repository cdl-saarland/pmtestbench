#include "instruction.h"

using namespace std;

unsigned Instruction::next_id = 0;

Instruction::Instruction(std::string_view name) : Name_(name) {
    id = next_id;
    next_id += 1;
}

const std::string &Instruction::getName(void) const {
    return Name_;
}

unsigned Instruction::getID(void) const {
    return id;
}

std::ostream& operator<< (std::ostream& stream, const Instruction& i) {
    stream << i.getName();
    return stream;
}

