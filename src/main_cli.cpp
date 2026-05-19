#include "core_logic.h"//the actual function
#include <iostream> //to print to terminal
#include <string> //for command line stuff
#include <cstdlib> //for return codes

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--config") {
        // 3 arguements should be "./caterpillar_cli --config config.json"
        std::cerr << "You have to pass: caterpillar_cli --config path/to/config.json\n";
        return EXIT_FAILURE;// apparently slurm jobs need this
        //CHANGE LATER if we want more flexibility
    }

    try {
        CoreLogic::runSimulationFromJson(argv[2]);
        return EXIT_SUCCESS;//also required for slurm jobs
    } catch (const std::exception& e) {
        std::cerr << "aww that didn't work: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}