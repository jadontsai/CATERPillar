#pragma once
#include "parameters.h"
#include "Axon.h"
#include "Blood_Vessel.h"
#include "Glial.h"
#include <string>
#include <tuple>
#include <vector>

class CoreLogic {
public:
    using SimResult = std::tuple<std::vector<Axon>, 
                                    std::vector<Blood_Vessel>, 
                                    std::vector<Glial>, 
                                    std::vector<Glial>>;

    // This function reads the JSON, fills the struct, and calls runSimulation()
    static void runSimulationFromJson(const std::string& jsonFilePath);
    static SimResult runSimulation(const Parameters& params);
};