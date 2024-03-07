#pragma once

#include "parser.h"
#include "../architecture.h"
#include "../experiment.h"
#include "../instruction.h"

#include <iostream>
#include <memory>
#include <vector>

class ExperimentParser: Parser {
public:
    ExperimentParser(std::istream &input, Architecture &arch, ExpVec &exp_set);

    bool parse(void);

private:
    void parseExperiment(void);
    void parseArchitecture(void);

    Architecture &arch_;
    ExpVec &exp_set_;
};

