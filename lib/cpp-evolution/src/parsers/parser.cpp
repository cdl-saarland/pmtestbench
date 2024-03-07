#include "parser.h"

#include <cctype>

Parser::Parser(std::istream &input) : input_(input) { }

bool Parser::isInputEmpty(void) {
    return input_empty_;
}

bool Parser::nextLine(void) {
    while (true) {
        if (! getLine()) {
            input_empty_ = true;
            return false;
        }
        if (curr_tokens.size() > 0) {
            return true;
        }
    }
}

void Parser::fillCurrTokens(void) {
    unsigned len = curr_line.size();
    unsigned start = 0;
    unsigned end = 0;
    unsigned i = 0;
    while (i < len) {
        while (isspace(curr_line[i])) {
            ++i;
            if (i >= len) {
                return;
            }
        }
        if (curr_line[i] == '"') {
            i += 1;
            start = i;
            while (curr_line[i] != '"') {
                ++i;
                if (i >= len) {
                    break;
                }
            }
            end = i;
            i += 1;
        } else {
            start = i;
            while (! isspace(curr_line[i])) {
                ++i;
                if (i >= len) {
                    break;
                }
            }
            end = i;
        }
        std::string_view token = curr_line;
        token.remove_prefix(start);
        token.remove_suffix(len - end);
        curr_tokens.push_back(token);
    }
}

bool Parser::getLine(void) {
    curr_tokens.clear();

    if (! std::getline(input_, curr_line)) {
        return false;
    }
    curr_line_no++;

    auto comment_idx = curr_line.find_first_of('#');
    // it's only a comment if the # is not preceeded by a non-space symbol
    while (comment_idx > 0 && comment_idx < curr_line.size() && !std::isspace(curr_line[comment_idx - 1])) {
        comment_idx = curr_line.find_first_of('#', comment_idx + 1);
    }

    if (comment_idx != std::string::npos) {
        curr_line.resize(comment_idx);
    }

    fillCurrTokens();

    return true;
}

void Parser::expect(bool cond, const std::string_view msg) {
    if (! cond) {
        flagError(msg);
    }
}

void Parser::nextLineOrFail(void) {
    expect(nextLine(), "File ended unexpectedly!");
}

void Parser::expectLine(std::string_view s) {
    expect(isLine(s), "Unexpected line!");
}

bool Parser::isLine(std::string_view s) {
    return (curr_tokens.size() == 1) && (curr_tokens[0] == s);
}

std::vector<std::string_view>& Parser::getCurrentLine(void) {
    return curr_tokens;
}

void Parser::flagError(std::string_view msg) {
    std::cerr << "Error in line " << curr_line_no << ": " << msg << "\n";
    std::cerr << "Offending line:\n";
    std::cerr << curr_line;
    std::cerr << std::endl;
    exit(EXIT_FAILURE);
}
