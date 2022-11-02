# Disable in-source builds to prevent source tree corruption
if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.11)

project(Sleipnir)

include(CPack)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# When building, don't use the install RPATH already (but later on when
# installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (isMultiConfig)
  if (NOT "Asan" IN_LIST CMAKE_CONFIGURATION_TYPES)
      list(APPEND CMAKE_CONFIGURATION_TYPES Asan)
  endif()
  if (NOT "Tsan" IN_LIST CMAKE_CONFIGURATION_TYPES)
      list(APPEND CMAKE_CONFIGURATION_TYPES Tsan)
  endif()
  if (NOT "Ubsan" IN_LIST CMAKE_CONFIGURATION_TYPES)
      list(APPEND CMAKE_CONFIGURATION_TYPES Ubsan)
  endif()
else()
  set(allowedBuildTypes Asan Tsan Ubsan Debug Release RelWithDebInfo MinSizeRel)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${allowedBuildTypes}")

  if (CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
  endif()
endif()

set(CMAKE_C_FLAGS_ASAN
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C compiler for Asan build type or configuration." FORCE)

set(CMAKE_CXX_FLAGS_ASAN
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C++ compiler for Asan build type or configuration." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_ASAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING
    "Linker flags to be used to create executables for Asan build type." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_ASAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING
    "Linker lags to be used to create shared libraries for Asan build type." FORCE)

set(CMAKE_C_FLAGS_TSAN
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=thread -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C compiler for Tsan build type or configuration." FORCE)

set(CMAKE_CXX_FLAGS_TSAN
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C++ compiler for Tsan build type or configuration." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_TSAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=thread" CACHE STRING
    "Linker flags to be used to create executables for Tsan build type." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_TSAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=thread" CACHE STRING
    "Linker lags to be used to create shared libraries for Tsan build type." FORCE)

set(CMAKE_C_FLAGS_UBSAN
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C compiler for Ubsan build type or configuration." FORCE)

set(CMAKE_CXX_FLAGS_UBSAN
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C++ compiler for Ubsan build type or configuration." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_UBSAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=undefined -fno-sanitize-recover=all" CACHE STRING
    "Linker flags to be used to create executables for Ubsan build type." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_UBSAN
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=undefined" CACHE STRING
    "Linker lags to be used to create shared libraries for Ubsan build type." FORCE)

file(GLOB_RECURSE Sleipnir_src src/*.cpp)

# Control where the static and shared libraries are built so that on Windows,
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)

add_library(Sleipnir ${Sleipnir_src})
if (NOT MSVC)
  target_compile_options(Sleipnir PRIVATE -Wall -pedantic -Wextra -Werror -Wno-unused-parameter)
else()
  target_compile_options(Sleipnir PRIVATE /wd4146 /wd4244 /wd4251 /wd4267 /WX)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set_target_properties(Sleipnir PROPERTIES DEBUG_POSTFIX "d")

set_property(TARGET Sleipnir PROPERTY FOLDER "libraries")
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)

target_compile_features(Sleipnir PUBLIC cxx_std_20)
if (MSVC)
  target_compile_options(Sleipnir PUBLIC /bigobj)
endif()

include(FetchContent)

FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)
target_link_libraries(Sleipnir Eigen3::Eigen)

option(FMT_INSTALL "Enable install for fmt project" ON)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        9.1.0
)
FetchContent_MakeAvailable(fmt)
target_link_libraries(Sleipnir fmt::fmt)

target_include_directories(Sleipnir PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/Sleipnir>)

install(TARGETS Sleipnir EXPORT Sleipnir DESTINATION lib)
install(DIRECTORY include/ DESTINATION "include/Sleipnir")

target_include_directories(Sleipnir
                           INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>)

install(EXPORT Sleipnir
  FILE Sleipnir.cmake
  DESTINATION lib/cmake/Sleipnir)

include(CMakePackageConfigHelpers)

# Generate the config file that is includes the exports
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Sleipnir-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/Sleipnir.cmake
  INSTALL_DESTINATION "lib/cmake/Sleipnir"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# Add benchmark executables if CasADi exists
find_package(casadi)
if (casadi_FOUND)
  file(GLOB FlywheelScalabilityBenchmark_src
    benchmarks/scalability/*.cpp
    benchmarks/scalability/flywheel/*.cpp)
  add_executable(FlywheelScalabilityBenchmark ${FlywheelScalabilityBenchmark_src})
  target_link_libraries(FlywheelScalabilityBenchmark PRIVATE Sleipnir casadi)

  file(GLOB CartPoleScalabilityBenchmark_src
    benchmarks/scalability/*.cpp
    benchmarks/scalability/cartpole/*.cpp)
  add_executable(CartPoleScalabilityBenchmark ${CartPoleScalabilityBenchmark_src})
  target_link_libraries(CartPoleScalabilityBenchmark PRIVATE Sleipnir casadi)
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
  CMAKE_ARGS     "-DCMAKE_BUILD_TYPE=ReleaseWithDebInfo"
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB_RECURSE Sleipnir_test_src test/*.cpp)
add_executable(SleipnirTest ${Sleipnir_test_src})
if (NOT MSVC)
  target_compile_options(SleipnirTest PRIVATE -Wall -pedantic -Wextra -Werror -Wno-unused-parameter)
else()
  target_compile_options(SleipnirTest PRIVATE /wd4146 /wd4244 /wd4251 /wd4267 /WX)
endif()
target_link_libraries(SleipnirTest PRIVATE Sleipnir GTest::gtest GTest::gtest_main)
add_test(
  NAME SleipnirTest
  COMMAND SleipnirTest
)
