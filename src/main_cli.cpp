//this file is the cli for both the cpu and gpu version

#include "core_logic.h"//the actual function
#include <iostream> //to print to terminal
#include <string> //for command line stuff
#include <cstdlib> //for return codes
#include <chrono>

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
    auto cpu_start = std::chrono::high_resolution_clock::now();

    try {
        CoreLogic::runSimulationFromJson(argv[2]);
        auto cpu_end = std::chrono::high_resolution_clock::now();

        double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        std::cout << "CPU simulation time_ms: " << cpu_ms << std::endl;
        return EXIT_SUCCESS;//also required for slurm jobs
    } catch (const std::exception& e) {
        std::cerr << "aww that didn't work: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    
}