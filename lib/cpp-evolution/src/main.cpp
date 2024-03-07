#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstring>

#include <unistd.h>

#include "parsers/config_parser.h"
#include "parsers/experiment_parser.h"
#include "parsers/mapping_parser.h"
#include "config.h"
#include "mapping.h"
#include "random_wrapper.h"
#include "runner.h"

#include "communicator.h"

using namespace std;

#ifndef DEFAULT_CFG_PATH
#define DEFAULT_CFG_PATH "run_configs/default.cfg"
#endif

static const char *const default_cfg_path = DEFAULT_CFG_PATH;

static const char *const command_file_path = "/tmp/pmevo-cmd";
static const char *const reply_file_path = "/tmp/pmevo-reply";

void print_usage(void) {
    std::cerr << "Usage:\n"
        << "  <PROGRAM> [options] <EXPERIMENTS>\n\n"
        << "Allowed options:\n"
        << "  -e <EXPS>        :  singleton experiments for all instructions\n"
        << "  -c <CONFIG>      :  read config file CONFIG (default: \"" << default_cfg_path << "\")\n"
        << "  -i               :  read experiments from stdin instead of file\n"
        << "  -j               :  print winners as json to stdout\n"
        << "  -m <MAPPING>     :  use the given mapping to evaluate the experiments\n"
        << "  -t <N>           :  print timing of evaluation as json to stderr, repeat experiments N times (only affects -m)\n"
        << "  -n <N>           :  print N best mappings after evoluation is finished (default: 1)\n"
        << "  -p <POPULATION>  :  read seed population of mappings from file POPULATION\n"
        << "  -x <JOURNAL>     :  write progress information to file JOURNAL if given, special values: 'stdout', 'stderr'\n"
        << "  -q <N>           :  override the number of ports given by the config\n"
        << "  -s <S>           :  seed for the random number generator (default: 424242)\n"
        << "\n"
        << "If executed in journaling mode (-x), write commands to "
        << command_file_path << " and find corresponding replies in "
        << reply_file_path << ". Try the 'help' command for possible commands.\n"
        ;
}

int main(int argc, char **const argv) {
    opterr = 0;
    const char *mapping_path = nullptr;
    const char *seed_population_path = nullptr;
    const char *journal_path = nullptr;
    const char *config_path = default_cfg_path;
    const char *exps_path = nullptr;
    const char *singleton_exp_path = nullptr;
    size_t num_mappings_to_print = 1;
    size_t cli_num_ports = 0;
    bool read_from_stdin = false;
    bool print_as_json = false;
    bool print_timing = false;
    unsigned timing_repetitions = 1;
    unsigned seed = 424242;
    int c = 0;

    while ((c = getopt (argc, argv, "c:m:p:x:n:q:s:t:e:ij")) != -1) {
        switch (c) {
        case 'c':
            config_path = optarg;
            break;
        case 'm':
            mapping_path = optarg;
            break;
        case 'p':
            seed_population_path = optarg;
            break;
        case 'x':
            journal_path = optarg;
            break;
        case 'e':
            singleton_exp_path = optarg;
            break;
        case 'i':
            read_from_stdin = true;
            break;
        case 'j':
            print_as_json = true;
            break;
        case 'n':
            try {
                num_mappings_to_print = stoul(string(optarg));
            } catch (exception& e) {
                std::cerr << "Invalid argument for option -n\n";
                return EXIT_FAILURE;
            }

            if (num_mappings_to_print < 1) {
                std::cerr << "Invalid argument for option -n: must be >0\n";
                return EXIT_FAILURE;
            }
            break;
        case 'q':
            try {
                cli_num_ports = stoul(string(optarg));
            } catch (exception& e) {
                std::cerr << "Invalid argument for option -q\n";
                return EXIT_FAILURE;
            }
            break;
        case 's':
            try {
                seed = stoul(string(optarg));
            } catch (exception& e) {
                std::cerr << "Invalid argument for option -s\n";
                return EXIT_FAILURE;
            }
            break;
        case 't':
            print_timing = true;
            try {
                timing_repetitions = stoul(string(optarg));
            } catch (exception& e) {
                std::cerr << "Invalid argument for option -t\n";
                return EXIT_FAILURE;
            }
            break;
        case '?':
            if (optopt == 'm' || optopt == 'c' || optopt == 'p' || optopt == 'x' || optopt == 'n' || optopt == 'q' || optopt == 's' || optopt == 't' || optopt == 'e') {
                std::cerr << "Option -" << optopt << " requires an argument.\n";
            } else {
                std::cerr << "Unknown option -" << optopt << ".\n";
            }
            print_usage();
            return EXIT_FAILURE;
        default:
            abort();
        }
    }
    if (optind >= argc && !read_from_stdin) {
        std::cerr << "Missing experiment file.\n";
        print_usage();
        return EXIT_FAILURE;
    }
    if (optind < argc - 1) {
        std::cerr << "Superfluous positional argument(s).\n";
        print_usage();
        return EXIT_FAILURE;
    }
    exps_path = argv[optind];

    Config config;

    ifstream configfile(config_path);
    ConfigParser(configfile, config).parse();

    size_t num_ports = cli_num_ports > 0 ? cli_num_ports : config.getNumPorts();

    ExpVec exp_set;
    Architecture arch(num_ports);

    istream *infile = nullptr;
    if (read_from_stdin) {
        infile = &cin;
    } else {
        infile = new ifstream{exps_path};
        if (! infile->good()) {
            cerr << "Error: Could not open experiment file '" << exps_path << "'!" << std::endl;
            return EXIT_FAILURE;
        }
    }
    ExperimentParser eparser(*infile, arch, exp_set);

    if (!eparser.parse()) {
        cerr << "Error while parsing input file" << std::endl;
        return EXIT_FAILURE;
    }

    if (mapping_path != nullptr) {
        ifstream mapinfile (mapping_path);
        if (! mapinfile.good()) {
            cerr << "Error: Could not open mapping file '" << mapping_path << "'!" << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<std::unique_ptr<Mapping>> mapping_set;

        MappingParser mparser(mapinfile, arch, mapping_set);

        if (!mparser.parse()) {
            cerr << "Error while parsing input file" << std::endl;
            return EXIT_FAILURE;
        }

        auto& mapping = *mapping_set[0];


        cout << "Simulating experiments with the following mapping:\n";
        cout << mapping << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        for (unsigned i = 0; i < timing_repetitions; ++i) {
            for (const auto& e : exp_set) {
                cout << "Simulating:\n" << *e;

                double res = mapping.simulateExperiment(arch, *e);

                cout << "result: " << res << "\n";

                if (print_timing) {
                    if (abs(res - e->getMeasuredCycles()) > 0.00001) {
                        cout << "Simulated result does not match measurement!\n";
                        return EXIT_FAILURE;
                    }
                }
            }
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;

        if (print_timing) {
            double total_secs = elapsed.count();
            double secs_per_exp = total_secs / (exp_set.size() * timing_repetitions);
            cerr << "{ \"total_secs\": " << elapsed.count() << ", \"secs_per_exp\": " << secs_per_exp << " }\n";
        }

        return EXIT_SUCCESS;
    }

    if (singleton_exp_path == nullptr) {
        cerr << "Error: -e parameter with path to singleton experiments missing\n";
        return EXIT_FAILURE;
    }
    istream *singleton_infile = nullptr;
    singleton_infile = new ifstream{singleton_exp_path};
    if (! singleton_infile->good()) {
        cerr << "Error: Could not open singleton experiment file '" << singleton_exp_path << "'!" << std::endl;
        return EXIT_FAILURE;
    }
    ExpVec singleton_exp_set;
    ExperimentParser singleton_eparser(*singleton_infile, arch, singleton_exp_set);

    if (!singleton_eparser.parse()) {
        cerr << "Error while parsing input file" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> singleton_results(arch.getInstructions().size());
    for (const auto &e : singleton_exp_set) {
        const auto &is = e->getInsnSeq();
        if (is.size() != 1) {
            cerr << "erroneous singleton experiment with more than one instruction" << std::endl;
            return EXIT_FAILURE;
        }
        const auto i = is[0];
        singleton_results.at(i->getID()) = e->getMeasuredCycles();
    }

    Mapping::EvalInfo eval_info{arch, exp_set, singleton_results, config};
    RandomWrapper rw{seed};

    const size_t population_size = config.getPopulationSize();

    std::vector<std::unique_ptr<Mapping>> init_mapping_set;
    if (seed_population_path != nullptr) {
        ifstream mapinfile (seed_population_path);
        MappingParser mparser(mapinfile, arch, init_mapping_set);

        if (!mparser.parse()) {
            cerr << "Error while parsing input file" << std::endl;
            return EXIT_FAILURE;
        }

        if (init_mapping_set.size() > population_size) {
            cerr << "Number of mappings in \"" << seed_population_path << "\" > PopulationSize" << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (journal_path != nullptr) {
        std::unique_ptr<ofstream> stream_ptr(nullptr);

        std::ostream *journal_stream = nullptr;
        if (strcmp(journal_path, "stdout") == 0) {
            journal_stream = &std::cout;
        } else if (strcmp(journal_path, "stderr") == 0) {
            journal_stream = &std::cerr;
        } else {
            stream_ptr.reset(new ofstream(journal_path));
            journal_stream = stream_ptr.get();
        }

        Communicator comm{command_file_path, reply_file_path};

        Runner<Mapping, true>runner(config, rw, eval_info, init_mapping_set, num_mappings_to_print, print_as_json, journal_stream, &comm);
        runner.doEvolution();
    } else {
        Runner<Mapping> runner(config, rw, eval_info, init_mapping_set, num_mappings_to_print, print_as_json);
        runner.doEvolution();
    }

    return EXIT_SUCCESS;
}

