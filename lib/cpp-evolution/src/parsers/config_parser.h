#pragma once

#include "parser.h"
#include "../architecture.h"
#include "../experiment.h"
#include "../instruction.h"

#include <iostream>
#include <memory>
#include <vector>

class ConfigParser: Parser {
public:
    ConfigParser(std::istream &input, Config &config);

    bool parse(void);

private:
    Config &config_;
};


