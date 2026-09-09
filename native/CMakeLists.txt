cmake_minimum_required(VERSION 3.16)
project(interpolate_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vulkan-headers/include" CACHE PATH "" FORCE)
find_library(INTERPOLATE_VULKAN_LIBRARY NAMES vulkan REQUIRED)
set(Vulkan_LIBRARY "${INTERPOLATE_VULKAN_LIBRARY}" CACHE FILEPATH "" FORCE)
find_package(Vulkan REQUIRED)

# Keep the native build small: the backend uses Vulkan inference only and does
# not need ncnn's tools, examples, tests, benchmarks, or OpenMP worker pool.
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_VULKAN_ONLINE_SPIRV ON CACHE BOOL "" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
set(NCNN_INT8 OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION ON CACHE BOOL "" FORCE)

add_subdirectory(vendor/ncnn ncnn EXCLUDE_FROM_ALL)

set(RIFE_SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rife")
set(RIFE_GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${RIFE_GENERATED_DIRECTORY}")

set(RIFE_SHADER_NAMES
    rife_preproc
    rife_postproc
    rife_preproc_tta
    rife_postproc_tta
    rife_flow_tta_avg
    rife_v2_flow_tta_avg
    rife_v4_flow_tta_avg
    rife_flow_tta_temporal_avg
    rife_v2_flow_tta_temporal_avg
    rife_v4_flow_tta_temporal_avg
    rife_out_tta_temporal_avg
    rife_v4_timestep
    rife_v4_timestep_tta
    warp
    warp_pack4
    warp_pack8
)

set(RIFE_GENERATED_HEADERS)
foreach(SHADER_NAME IN LISTS RIFE_SHADER_NAMES)
    set(SHADER_SOURCE "${RIFE_SOURCE_DIRECTORY}/${SHADER_NAME}.comp")
    set(SHADER_HEADER "${RIFE_GENERATED_DIRECTORY}/${SHADER_NAME}.comp.hex.h")
    add_custom_command(
        OUTPUT "${SHADER_HEADER}"
        COMMAND "${CMAKE_COMMAND}"
            -DSHADER_SRC=${SHADER_SOURCE}
            -DSHADER_COMP_HEADER=${SHADER_HEADER}
            -P "${RIFE_SOURCE_DIRECTORY}/generate_shader_comp_header.cmake"
        DEPENDS "${SHADER_SOURCE}" "${RIFE_SOURCE_DIRECTORY}/generate_shader_comp_header.cmake"
        VERBATIM
    )
    list(APPEND RIFE_GENERATED_HEADERS "${SHADER_HEADER}")
endforeach()

add_library(interpolate_backend STATIC
    interpolate_backend.cpp
    rife/rife.cpp
    rife/warp.cpp
    ${RIFE_GENERATED_HEADERS}
)
# ncnn disables exceptions through an interface option. Re-enable them only in
# the ABI wrapper, which catches every exception before returning to Rust.
set_source_files_properties(interpolate_backend.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")

target_include_directories(interpolate_backend
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${RIFE_SOURCE_DIRECTORY}"
        "${RIFE_GENERATED_DIRECTORY}"
)
target_link_libraries(interpolate_backend PRIVATE ncnn Vulkan::Vulkan Threads::Threads)
target_compile_options(interpolate_backend PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fexceptions -UNDEBUG -Wall -Wextra -Wpedantic>
)
set_target_properties(interpolate_backend PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
