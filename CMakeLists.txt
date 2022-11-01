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
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Sleipnir-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/Sleipnir.cmake"
  INSTALL_DESTINATION "lib/cmake/Sleipnir"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
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
