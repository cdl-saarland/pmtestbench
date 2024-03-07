#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

class Parser {
protected:
    Parser(std::istream &input);

    bool isInputEmpty(void);

    bool nextLine(void);

    void expect(bool cond, const std::string_view msg);

    void nextLineOrFail(void);

    void expectLine(std::string_view s);

    bool isLine(std::string_view s);

    std::vector<std::string_view>& getCurrentLine(void);

    void flagError[[noreturn]](std::string_view msg);

private:
    bool getLine(void);

    void fillCurrTokens(void);

    unsigned curr_line_no = 0;
    std::string curr_line;
    std::vector<std::string_view> curr_tokens;
    std::istream& input_;
    bool input_empty_ = false;
};

