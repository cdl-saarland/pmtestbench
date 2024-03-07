#include "communicator.h"


#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>

#define PREP_TIME \
    auto now = std::chrono::system_clock::now(); \
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

#define TIME "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << "]"

Communicator::Communicator(std::string cmdfile, std::string replyfile, std::vector<CommandSpec> &&cmds) : filename(cmdfile), replyfilename(replyfile), commands(cmds) {
    clearFile(this->filename);
    clearFile(this->replyfilename);
    this->registerCommand({"help", [&](auto& os, auto){
            os << "Available commands:\n";
            for (const auto & [str, fn] : this->commands) {
                os << "  " << str << std::endl;
            }
        }});
}

Communicator::Communicator(std::string cmdfile, std::string replyfile) : Communicator(cmdfile, replyfile, {}) { }

void Communicator::clearFile(const std::string &fn) {
    std::ofstream outfile;
    outfile.open(fn, std::ios::out | std::ios::trunc);
    PREP_TIME;
    outfile << "ready for command " << TIME << std::endl;
    outfile.close();
}

void Communicator::registerCommand(CommandSpec cmd) {
    commands.push_back(cmd);
}

void Communicator::checkCommands(void) {
    std::ifstream infile;
    infile.open(this->filename, std::ios::in);

    std::ofstream outfile;
    outfile.open(this->replyfilename, std::ios::out | std::ios::app);

    std::string line;
    while (getline(infile, line)) {
        if (line.rfind("ready for command", 0) == 0) { // starts_with
            break;
        }
        PREP_TIME;
        outfile << TIME << " start handling command '" << line << "'" << std::endl;
        bool done = false;
        for (const auto &[cmd, func] : commands) {
            if (line.rfind(cmd, 0) == 0) {
                func(outfile, line);
                done = true;
                break;
            }
        }
        if (! done) {
            outfile << "  No such command!" << std::endl;
        }
        {
            PREP_TIME;
            outfile << TIME << " done handling command '" << line << "'" << std::endl;
        }
    }

    infile.close();
    outfile.close();

    clearFile(this->filename);
}

