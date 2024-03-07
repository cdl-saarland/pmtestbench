#include "mapping_parser.h"

#include <cassert>
#include <string>
#include <string_view>

using namespace std;

MappingParser::MappingParser(istream &input, Architecture &arch, MappingSet &mset) : Parser(input), arch_(arch), mappings_(mset) { }

bool MappingParser::parse(void) {
    nextLineOrFail();
    parseMapping();
    while (! isInputEmpty()) {
        parseMapping();
    }

    // TODO check that there is a mapping for all instructions

    return true;
}

void MappingParser::parseMapping(void) {
    auto &currentLineTokens = getCurrentLine();

    expect(currentLineTokens.size() == 1, "Expected 'mapping:' line!");
    expect(currentLineTokens[0] == "mapping:", "Expected 'mapping:' line!");
    nextLineOrFail();

    auto mapping = new Mapping();

    while (! isInputEmpty()) {
        parseInsn(*mapping);
        if (! isInputEmpty()) {
            assert(currentLineTokens.size() == 1);
            if (currentLineTokens[0] == "mapping:") {
                break;
            }
        }
    }
    mappings_.emplace_back(mapping);
}

Uop MappingParser::strToUop(std::string_view in) {

    Uop res = 0;

    for (char c : in) {
        if (c < 'A' || c > 'Z') {
            flagError("Invalid port name in uop line!");
        }
        Uop mask = 1 << (c - 'A');
        if (mask & res) {
            flagError("Duplicate port name in uop line!");
        }
        res |= mask;
    }

    return res;
}

void MappingParser::parseInsn(Mapping &mapping) {
    auto &currentLineTokens = getCurrentLine();
    expect(currentLineTokens.size() == 1, "Invalid instruction line!");
    expect(currentLineTokens[0].back() == ':', "Missing terminating colon ':' in instruction line!");
    string_view insn_name = currentLineTokens[0].substr(0, currentLineTokens[0].size() - 1);

    Instruction *insn = arch_.getInstruction(insn_name);
    mapping.addInsn(insn);

    while (nextLine()) {
        if (currentLineTokens.size() == 1) {
            break;
        }
        expect(currentLineTokens.size() == 2, "Invalid uop line!");
        expect(currentLineTokens[0].back() == ':', "Missing colon ':' in uop line!");

        Uop uop = strToUop(currentLineTokens[0].substr(0, currentLineTokens[0].size() - 1));

        if (uop & ~arch_.getLargestUop()) {
            flagError("Mapping uses uop that is not in specified architecture!");
        }

        unsigned num = 0;
        try {
            num = stoi(string(currentLineTokens[1]));
        } catch (exception& e) {
            flagError("Invalid uop number!");
        }

        if (! mapping.addEntry(insn, uop, num)) {
            flagError("Duplicate uop entry!");
        }
    }
}
