#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

/// This class implements a rudimentary interface for inter-process
/// (or human-to-process) communication.
///
/// Communication takes place via two files, a command file that is read an
/// checked for commands whenever the checkCommands() method is called and the
/// reply file in which the communicator writes replies to the commands.
///
/// Commands are specified dynamically via the registerCommand() method. They
/// consisit of a string that is checked for in the command file (the command
/// string needs to be a prefix of the line in the command file) and a function
/// that is called when the corresponding command is recognized. It is invoked
/// with a string including the full matching line of the command file and an
/// output stream for writing replies into the reply file.
///
/// Commands are handled in bulk whenever checkCommands() is called, afterwards
/// the command file is cleared. The reply file is cleared on creation of the
/// Communicator object, all subsequent writes to it are appending.
class Communicator {
public:
    using CommandSpec = std::pair<std::string, std::function<void(std::ostream&, std::string_view)>>;

    Communicator(std::string cmdfile, std::string replyfile);

    Communicator(std::string cmdfile, std::string replyfile, std::vector<CommandSpec> &&cmds);

    void registerCommand(CommandSpec cmd);

    void checkCommands(void);

private:
    const std::string filename;
    const std::string replyfilename;

    void clearFile(const std::string &fn);

    std::vector<CommandSpec> commands;
};

