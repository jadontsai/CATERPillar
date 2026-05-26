cmake_minimum_required(VERSION 3.10)
project(CATERPillar VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
#required for std::filesystem
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CATERPILLAR_ENABLE_CUDA "Enable CUDA GPU backend" OFF)

find_package(Eigen3 REQUIRED)
#only for CPU; just check For GPU version later

add_library(caterpillar_core
    src/Axon.cpp
    src/Blood_Vessel.cpp
    src/Glial.cpp
    src/axongammadistribution.cpp
    src/core_logic.cpp
    src/grow_axons.cpp
    src/grow_blood_vessels.cpp
    src/grow_cells.cpp
    src/grow_glial_cells.cpp
    src/obstacle.cpp
    src/sphere.cpp
    src/threads.cpp
)

target_include_directories(caterpillar_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
#I think i can keep this public? may want to change later
target_link_libraries(caterpillar_core PUBLIC 
Eigen3::Eigen)

add_executable(caterpillar_cli
    src/main_cli.cpp
)

target_link_libraries(caterpillar_cli PRIVATE 
caterpillar_core)


#CUDA stuff
if(CATERPILLAR_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(caterpillar_gpu
        src/gpu_stuff/gpu_simulation_state.cu
        src/gpu_stuff/gpu_runner.cu
        src/gpu_stuff/gpu_init.cu
        src/gpu_stuff/gpu_candidate_generation.cu
        src/gpu_stuff/gpu_in_box.cu
        src/gpu_stuff/gpu_collision_check.cu
        src/gpu_stuff/gpu_commit.cu
        src/gpu_stuff/gpu_spatial_grid.cu
        src/gpu_stuff/gpu_candidate_selection.cu
        src/gpu_stuff/gpu_util.cu
    )#9 files

    target_include_directories(caterpillar_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_stuff
    )

    target_link_libraries(caterpillar_gpu PUBLIC 
    caterpillar_core)

    target_compile_definitions(caterpillar_gpu PUBLIC 
    CATERPILLAR_ENABLE_CUDA=1)

    target_compile_definitions(caterpillar_cli PRIVATE 
    CATERPILLAR_ENABLE_CUDA=1)

    target_link_libraries(caterpillar_cli PRIVATE 
    caterpillar_gpu)

    set_target_properties(caterpillar_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "60;70;80;86;90"
    )
endif()