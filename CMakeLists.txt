cmake_minimum_required(VERSION 3.24)

project(Sleipnir)

include(GoogleTest)

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

file(GLOB_RECURSE Sleipnir_src src/*.cpp)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
add_library(Sleipnir ${Sleipnir_src})
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
set_target_properties(Sleipnir PROPERTIES DEBUG_POSTFIX "d")

set_property(TARGET Sleipnir PROPERTY FOLDER "libraries")
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)

target_compile_features(Sleipnir PUBLIC cxx_std_20)
if (MSVC)
    target_compile_options(Sleipnir PUBLIC /bigobj)
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
target_link_libraries(Sleipnir Eigen3::Eigen fmt::fmt)

target_include_directories(Sleipnir PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${include_dest}/Sleipnir>)

install(TARGETS Sleipnir EXPORT Sleipnir DESTINATION "${main_lib_dest}")
install(DIRECTORY src/main/native/include/ DESTINATION "${include_dest}/Sleipnir")

file(GLOB_RECURSE Sleipnir_test_src test/*.cpp)
add_executable(SleipnirTest ${Sleipnir_test_src})
gtest_discover_tests(SleipnirTest)
target_link_libraries(SleipnirTest PRIVATE Sleipnir gtest gtest_main)
