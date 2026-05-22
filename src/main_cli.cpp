#include "core_logic.h"//the actual function
#include <iostream> //to print to terminal
#include <string> //for command line stuff
#include <cstdlib> //for return codes

#ifdef CATERPILLAR_ENABLE_CUDA//if cuda, then use cuda (self explanatory?)
#include "gpu_runner.h"
#endif
int main(int argc, char** argv) {
    #ifdef CATERPILLAR_ENABLE_CUDA
    //for gpu testing, will be removed later
    if (argc == 2 && std::string(argv[1]) == "--gpu-smoke-test") {
        GpuParameters params;
        run_gpu_simulation(params);
        return EXIT_SUCCESS;
    }
    #endif
    if (argc < 3 || std::string(argv[1]) != "--config") {
        //cpu command line interface version
        // 3 arguements should be "./caterpillar_cli --config config.json"
        std::cerr << "You have to pass: caterpillar_cli --config path/to/config.json\n";
        return EXIT_FAILURE;
    }

    try {
        CoreLogic::runSimulationFromJson(argv[2]);
        return EXIT_SUCCESS;//also required for slurm jobs
    } catch (const std::exception& e) {
        std::cerr << "aww that didn't work: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}