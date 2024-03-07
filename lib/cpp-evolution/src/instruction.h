#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

class Experiment;

class Instruction {
public:
    explicit Instruction(std::string_view name);

    const std::string &getName(void) const;

    unsigned getID(void) const;

    friend std::ostream& operator<< (std::ostream& stream, const Instruction& i);

private:
    std::string Name_;
    unsigned id;

    static unsigned next_id;
};

