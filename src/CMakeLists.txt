# Spatter/src/CMakeLists.txt

add_subdirectory(Spatter)

add_executable(spatter_in_compute_core main.cc)
#target_compile_options(spatter_in_compute_core PUBLIC "-fnew-alignment 32")

target_include_directories(
            spatter_in_compute_core
            PRIVATE
                ${UMD_HOME}
                ${PROJECT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/tt_metal
                ${PROJECT_SOURCE_DIR}/tt_metal/common
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/tt_metal/programming_examples/spatter_in_compute/src/Spatter
        )
#                ${PROJECT_SOURCE_DIR}/tt_metal/programming_examples/spatter_in_compute/build_serial/_deps/nlohmann_json-src/include

target_link_libraries(
            spatter_in_compute_core
            PUBLIC
                tt_metal
                m
                pthread
                Spatter
        )
set_target_properties(
            spatter_in_compute_core
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${PROJECT_BINARY_DIR}
        )

list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS spatter_in_compute_core)

add_custom_target(spatter_in_compute DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})