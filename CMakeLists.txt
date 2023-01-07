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

include(SleipnirBuildTypes)

# Control where the static and shared libraries are built so that on Windows,
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)

include(SleipnirCompilerFlags)

file(GLOB_RECURSE Sleipnir_src src/*.cpp thirdparty/llvm/src/*.cpp)
add_library(Sleipnir ${Sleipnir_src})
add_library(Sleipnir::Sleipnir ALIAS Sleipnir)
sleipnir_compiler_flags(Sleipnir)
target_include_directories(Sleipnir
                           PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llvm/include)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set_target_properties(Sleipnir PROPERTIES DEBUG_POSTFIX "d")

set_property(TARGET Sleipnir PROPERTY FOLDER "libraries")
target_compile_definitions(Sleipnir PRIVATE SLEIPNIR_EXPORTS)

include(FetchContent)

# Eigen dependency
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)

target_link_libraries(Sleipnir PUBLIC Eigen3::Eigen)

# Prevent Eigen tests from running
add_custom_command(
  TARGET Sleipnir
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/_deps/eigen3-build/CTestTestfile.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# fmt dependency
set(FMT_INSTALL ON)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        9.1.0
)
FetchContent_MakeAvailable(fmt)

target_link_libraries(Sleipnir PRIVATE fmt)

target_include_directories(Sleipnir
                           PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/Sleipnir>)

install(TARGETS Sleipnir
        EXPORT SleipnirTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
export(TARGETS Sleipnir
       FILE Sleipnir.cmake
       NAMESPACE Sleipnir::)
install(DIRECTORY include/ DESTINATION "include/Sleipnir")
install(EXPORT SleipnirTargets
        FILE Sleipnir.cmake
        NAMESPACE Sleipnir::
        DESTINATION lib/cmake/Sleipnir)

include(CMakePackageConfigHelpers)

# Generate the config file that includes the exports
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SleipnirConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfig.cmake
  INSTALL_DESTINATION "lib/cmake/Sleipnir"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# Install the config file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SleipnirConfig.cmake
  DESTINATION lib/cmake/Sleipnir)

option(BUILD_BENCHMARKING "Build CasADi and Sleipnir benchmarks" ON)

# Add benchmark executables if CasADi exists
if (BUILD_BENCHMARKING)
  find_package(casadi QUIET)
  if (casadi_FOUND)
    foreach (benchmark "CartPole" "Flywheel")
      file(GLOB ${benchmark}ScalabilityBenchmark_src
        benchmarks/scalability/*.cpp
        benchmarks/scalability/${benchmark}/*.cpp)
      add_executable(${benchmark}ScalabilityBenchmark
                     ${${benchmark}ScalabilityBenchmark_src})
      sleipnir_compiler_flags(${benchmark}ScalabilityBenchmark)
      target_include_directories(${benchmark}ScalabilityBenchmark
                                 PRIVATE
                                 ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/scalability
                                 ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/units/include)
      target_link_libraries(${benchmark}ScalabilityBenchmark
                            PRIVATE
                            Sleipnir
                            fmt
                            casadi)
    endforeach()
  endif()
endif()

# GoogleTest dependency (static linkage)
set(BUILD_SHARED_LIBS_SAVE ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
option(INSTALL_GMOCK "Install GoogleTest's GMock" OFF)
option(INSTALL_GTEST "Install GoogleTest's GTest" OFF)
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

# Build Sleipnir tests
if (BUILD_TESTING)
  file(GLOB_RECURSE Sleipnir_test_src test/src/*.cpp)
  add_executable(SleipnirTest ${Sleipnir_test_src})
  sleipnir_compiler_flags(SleipnirTest)
  target_include_directories(SleipnirTest
                             PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/src
                             ${CMAKE_CURRENT_SOURCE_DIR}/test/include
                             ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/units/include)
  target_link_libraries(SleipnirTest
                        PRIVATE
                        Sleipnir
                        fmt
                        GTest::gtest
                        GTest::gtest_main)
  if (NOT CMAKE_TOOLCHAIN_FILE)
    gtest_discover_tests(SleipnirTest)
  endif()
endif()

# Build examples and example tests
include(SleipnirSubdirList)
sleipnir_subdir_list(EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples)
foreach(example ${EXAMPLES})
  # Build example
  file(GLOB_RECURSE sources examples/${example}/src/*.cpp)
  add_executable(${example} ${sources})
  sleipnir_compiler_flags(${example})
  target_include_directories(${example}
                             PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/include
                             ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/units/include)
  target_link_libraries(${example}
                        PRIVATE
                        Sleipnir
                        fmt)

  # Build example test if files exist for it
  if (BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/test)
    file(GLOB_RECURSE test_sources examples/${example}/test/*.cpp)
    add_executable(${example}Test ${sources} ${test_sources})
    target_include_directories(${example}Test
                               PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/src
                               ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/test
                               ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/units/include)
    sleipnir_compiler_flags(${example}Test)
    target_compile_definitions(${example}Test PUBLIC RUNNING_TESTS)
    target_include_directories(${example}Test
                               PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/units/include)
    target_link_libraries(${example}Test
                          PRIVATE
                          Sleipnir
                          fmt
                          GTest::gtest
                          GTest::gtest_main)
    if (NOT CMAKE_TOOLCHAIN_FILE)
      gtest_discover_tests(${example}Test)
    endif()
  endif()
endforeach()
