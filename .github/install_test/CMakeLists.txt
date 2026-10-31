cmake_minimum_required(VERSION 3.22)
project(SleipnirInstallTest LANGUAGES CXX)

find_package(Sleipnir CONFIG REQUIRED)

add_executable(install_test main.cpp)
target_compile_features(install_test PRIVATE cxx_std_23)
target_link_libraries(install_test PRIVATE Sleipnir::Sleipnir)

enable_testing()
add_test(NAME install_test COMMAND install_test)

# Windows needs the Sleipnir DLL on PATH
set_tests_properties(
    install_test
    PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:Sleipnir::Sleipnir>"
)
