# CMakeLists.txt
#set(CMAKE_PREFIX_PATH "/home/user/cmake_3_30_6_version/cmake-3.30.6-build")

set(CMAKE_SOURCE_DIR "/home/user/tt-metal/tt_metal/programming_examples/spatter_in_compute")
cmake_minimum_required(VERSION 3.25)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(DisableIntree)

project(Spatter VERSION 2.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

include(BuildType)
include(CompilerType)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED on)
set(USE_OPENMP off)
set(USE_MPI off)
include(pkgs/JSONSupport)
include(pkgs/MPISupport)
include(pkgs/OpenMPSupport)
include(pkgs/CUDASupport)

# Create gz_read executable
add_executable(gz_read standard-suite/binary-traces/gz_read.cc)

#target_link_libraries(gz_read z)

target_link_libraries(
            gz_read
            PUBLIC
                tt_metal
                m
                pthread
                z
        )


target_include_directories(
            gz_read
            PRIVATE
                ${UMD_HOME}
                ${PROJECT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/tt_metal
                ${PROJECT_SOURCE_DIR}/tt_metal/common
                ${CMAKE_CURRENT_SOURCE_DIR}
        )

message("${UMD_HOME}")
message("${PROJECT_SOURCE_DIR}")
message("${CMAKE_CURRENT_SOURCE_DIR}")
message("${PROJECT_BINARY_DIR}")

set_target_properties(
            gz_read
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${PROJECT_BINARY_DIR}
        )

list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS  gz_read)

add_subdirectory(src)

#enable_testing()
#add_subdirectory(tests)

#set(VECADD_SRCS1 /home/user/tt-metal/tt_metal/programming_examples/spatter/standard-suite/binary-traces/gz_read.cc)

#CREATE_PGM_EXAMPLES_EXE("${VECADD_SRCS1}" "spatter") # output binaries to build/programming_examples/contributed

#add_custom_target(spatter_in_compute DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})