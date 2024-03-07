#pragma once

#include "parser.h"
#include "../architecture.h"
#include "../instruction.h"
#include "../mapping.h"

#include <iostream>
#include <vector>

using MappingSet = std::vector<std::unique_ptr<Mapping>>;

// TODO refactor for reasonable ownership management, cf. exp parser
class MappingParser: Parser {
public:
    MappingParser(std::istream &input, Architecture &arch, MappingSet &mset);

    bool parse(void);

private:
    void parseInsn(Mapping &mapping);

    void parseMapping(void);

    Uop strToUop(std::string_view in);

    Architecture &arch_;
    MappingSet &mappings_;
};



