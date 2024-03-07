#include "experiment_parser.h"

#include <cassert>
#include <string>

using namespace std;

ExperimentParser::ExperimentParser(
        istream &input,
        Architecture &arch,
        ExpVec &exp_set)
    : Parser(input),
        arch_(arch),
        exp_set_(exp_set) { }


bool ExperimentParser::parse(void) {
    nextLineOrFail();
    parseArchitecture();
    do  {
        parseExperiment();
    } while (nextLine());

    return true;
}


void ExperimentParser::parseExperiment(void) {
    // ends on "cycle" line of experiment entry
    expectLine("experiment:");
    nextLineOrFail();
    expectLine("instructions:");
    nextLineOrFail();
    std::vector<Instruction*> current_insn_list;

    auto &currentLineTokens = getCurrentLine();

    while (! (currentLineTokens[0] == "cycles:")) {
        expect(currentLineTokens[0].back() != ':', "Invalid instruction line!");
        current_insn_list.push_back(arch_.getInstruction(currentLineTokens[0]));
        nextLineOrFail();
    }
    expect(currentLineTokens.size() == 2, "Invalid 'cycles' line!");
    double cycles = 0.0;
    try {
        cycles = stof(string(currentLineTokens[1]));
    } catch (exception& e) {
        flagError("Invalid cycle number!");
    }

    auto new_exp = std::unique_ptr<Experiment>(new Experiment(current_insn_list));
    new_exp->setMeasuredCycles(cycles);
    exp_set_.push_back(std::move(new_exp));
}


void ExperimentParser::parseArchitecture(void) {
    expectLine("architecture:");
    nextLineOrFail();
    expectLine("instructions:");
    nextLineOrFail();

    auto &currentLineTokens = getCurrentLine();

    while (! (currentLineTokens[0] == "experiment:")) {
        expect(currentLineTokens[0].back() != ':', "Invalid instruction line!");
        arch_.getInstruction(currentLineTokens[0]);
        nextLineOrFail();
    }
}
