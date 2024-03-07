#include "config_parser.h"

#include <cassert>
#include <string>

using namespace std;

ConfigParser::ConfigParser(
        istream &input,
        Config &config)
    : Parser(input),
        config_(config) { }


#define CHECKTOKEN(NAME, RES_T, CONVERT_FUN)                                   \
        if (token == #NAME ":") {                                              \
            expect(currentLineTokens.size() == 2, "Invalid config option!");   \
            RES_T res;                                                         \
            try {                                                              \
                res = CONVERT_FUN(string(currentLineTokens[1]));               \
            } catch (exception& e) {                                           \
                flagError("Invalid value!");                                   \
            }                                                                  \
            config_.NAME = res;                                                \
        }

namespace {
    bool stobool(const string &s) {
        if (s == "True" || s == "true" || s == "1") {
            return true;
        }
        if (s == "False" || s == "false" || s == "0") {
            return false;
        }
        throw new exception();
    }
}

bool ConfigParser::parse(void) {
    nextLineOrFail();
    auto &currentLineTokens = getCurrentLine();
    expectLine("configuration:");
    while (nextLine()) {
        expect(currentLineTokens.size() >= 2, "Invalid config option!");
        auto token = currentLineTokens[0];
        expect(token.back() == ':', "Missing colon in config option!");

        CHECKTOKEN(PopulationSize, int, stoi)
        else
        CHECKTOKEN(MaxRecombinationFactor, double, stof)
        else
        CHECKTOKEN(MaxMutationFactor, double, stof)
        else
        CHECKTOKEN(NumIterations, int, stoi)
        else
        CHECKTOKEN(NumEpochs, int, stoi)
        else
        CHECKTOKEN(KeepRatio, double, stof)
        else
        CHECKTOKEN(NumPorts, int, stoi)
        else
        CHECKTOKEN(LuckChance, double, stof)
        else
        CHECKTOKEN(MutAddUopChance, double, stof)
        else
        CHECKTOKEN(MutChangeUopChance, double, stof)
        else
        CHECKTOKEN(MutChangeNumChance, double, stof)
        else
        CHECKTOKEN(BadLuckProtection, double, stof)
        else
        CHECKTOKEN(EnableLocalOptimization, bool, stobool)
        else
        CHECKTOKEN(EnableRatioCombination, bool, stobool)
        else
        CHECKTOKEN(EnableSymmetricError, bool, stobool)
        else
        CHECKTOKEN(PeakIPC, double, stof)
    }

    return true;
}

