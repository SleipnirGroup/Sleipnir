# Disable in-source builds to prevent source tree corruption
if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.21)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(Sleipnir)

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

# Generate compile_commands.json by default
if (NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
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
  if (NOT "Perf" IN_LIST CMAKE_CONFIGURATION_TYPES)
      list(APPEND CMAKE_CONFIGURATION_TYPES Perf)
  endif()
else()
  set(allowedBuildTypes Asan Tsan Ubsan Perf Debug Release RelWithDebInfo MinSizeRel)
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

set(CMAKE_C_FLAGS_PERF
    "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C compiler for perf build type or configuration." FORCE)

set(CMAKE_CXX_FLAGS_PERF
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer" CACHE STRING
    "Flags used by the C++ compiler for perf build type or configuration." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_PERF
    "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}" CACHE STRING
    "Linker flags to be used to create executables for perf build type." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_PERF
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" CACHE STRING
    "Linker lags to be used to create shared libraries for perf build type." FORCE)

# Control where the static and shared libraries are built so that on Windows,
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)

include(SleipnirCompilerFlags)

file(GLOB_RECURSE Sleipnir_src src/*.cpp)
add_library(Sleipnir ${Sleipnir_src})
sleipnir_compiler_flags(Sleipnir)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set_target_properties(Sleipnir PROPERTIES DEBUG_POSTFIX "d")

set_property(TARGET Sleipnir PROPERTY FOLDER "libraries")
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)

target_compile_features(Sleipnir PUBLIC cxx_std_20)
if (MSVC)
  target_compile_options(Sleipnir PUBLIC /bigobj)
endif()

include(FetchContent)

# Eigen dependency (static linkage)
set(BUILD_SHARED_LIBS_SAVE ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVE})

target_link_libraries(Sleipnir PUBLIC Eigen3::Eigen)

# Prevent Eigen tests from running
add_custom_command(
  TARGET Sleipnir
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_BINARY_DIR}/_deps/eigen3-build/CTestTestfile.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# fmt dependency (static linkage)
set(BUILD_SHARED_LIBS_SAVE ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        9.1.0
)
FetchContent_MakeAvailable(fmt)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVE})

target_link_libraries(Sleipnir PRIVATE fmt)

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
find_package(casadi QUIET)
if (casadi_FOUND)
  file(GLOB FlywheelScalabilityBenchmark_src
    benchmarks/scalability/*.cpp
    benchmarks/scalability/flywheel/*.cpp)
  add_executable(FlywheelScalabilityBenchmark ${FlywheelScalabilityBenchmark_src})
  sleipnir_compiler_flags(FlywheelScalabilityBenchmark)
  target_link_libraries(FlywheelScalabilityBenchmark PRIVATE Sleipnir fmt casadi)

  file(GLOB CartPoleScalabilityBenchmark_src
    benchmarks/scalability/*.cpp
    benchmarks/scalability/cartpole/*.cpp)
  add_executable(CartPoleScalabilityBenchmark ${CartPoleScalabilityBenchmark_src})
  sleipnir_compiler_flags(CartPoleScalabilityBenchmark)
  target_link_libraries(CartPoleScalabilityBenchmark PRIVATE Sleipnir fmt casadi)
endif()

# GoogleTest dependency (static linkage)
set(BUILD_SHARED_LIBS_SAVE ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
  CMAKE_ARGS     "-DCMAKE_BUILD_TYPE=ReleaseWithDebInfo"
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVE})

cmake_dependent_option(BUILD_TESTING "Enable creation of tests." ON "PROJECT_IS_TOP_LEVEL" OFF)
if (BUILD_TESTING)
  enable_testing()
  include(GoogleTest)
endif()

# Build examples and their tests
include(SleipnirSubdirList)
sleipnir_subdir_list(EXAMPLES ${CMAKE_SOURCE_DIR}/examples)
foreach(example ${EXAMPLES})
  file(GLOB_RECURSE sources examples/${example}/src/*.cpp)
  add_executable(${example} ${sources})
  sleipnir_compiler_flags(${example})
  target_include_directories(${example} PUBLIC examples/${example}/include)
  target_link_libraries(${example} PRIVATE Sleipnir fmt)

  if (BUILD_TESTING AND EXISTS ${CMAKE_SOURCE_DIR}/examples/${example}/test)
    file(GLOB_RECURSE test_sources examples/${example}/test/*.cpp)
    add_executable(${example}Test ${sources} ${test_sources})
    target_include_directories(${example}Test PRIVATE
      examples/${example}/src
      examples/${example}/test)
    sleipnir_compiler_flags(${example}Test)
    target_compile_definitions(${example}Test PUBLIC RUNNING_TESTS)
    target_link_libraries(${example}Test PRIVATE
      Sleipnir fmt GTest::gtest GTest::gtest_main)
    gtest_discover_tests(${example}Test)
  endif()
endforeach()

if (BUILD_TESTING)
  file(GLOB_RECURSE Sleipnir_test_src test/*.cpp)
  add_executable(SleipnirTest ${Sleipnir_test_src})
  sleipnir_compiler_flags(SleipnirTest)
  target_link_libraries(SleipnirTest PRIVATE
    Sleipnir fmt GTest::gtest GTest::gtest_main)
  gtest_discover_tests(SleipnirTest)
endif()
