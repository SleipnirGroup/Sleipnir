function(pybind11_mkdoc target headers)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Get default clang version
    execute_process(
        COMMAND clang++ --version
        OUTPUT_VARIABLE CLANG_VERSION
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(REGEX MATCH "[0-9]+" CLANG_VERSION ${CLANG_VERSION})

    if(UNIX AND NOT APPLE)
        if(EXISTS /usr/lib/libclang.so)
            set(env_vars
                LLVM_DIR_PATH=/usr/lib
                LIBCLANG_PATH=/usr/lib/libclang.so
            )
        else()
            set(env_vars
                LLVM_DIR_PATH=/usr/lib/llvm-${CLANG_VERSION}
                LIBCLANG_PATH=/usr/lib/llvm-${CLANG_VERSION}/lib/libclang.so
            )
        endif()
    endif()

    get_target_property(target_dirs ${target} INCLUDE_DIRECTORIES)
    list(TRANSFORM target_dirs PREPEND "-I")

    get_target_property(eigen_dirs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    list(FILTER eigen_dirs INCLUDE REGEX "\\$<BUILD_INTERFACE:.*>")
    list(TRANSFORM eigen_dirs PREPEND "-I")

    get_target_property(
        small_vector_dirs
        small_vector
        INTERFACE_INCLUDE_DIRECTORIES
    )
    list(FILTER small_vector_dirs INCLUDE REGEX "\\$<BUILD_INTERFACE:.*>")
    list(TRANSFORM small_vector_dirs PREPEND "-I")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
        COMMAND
            ${env_vars} ${Python3_EXECUTABLE} -m pybind11_mkdoc ${headers} -o
            ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
            -I/usr/lib/clang/${CLANG_VERSION}/include ${target_dirs}
            ${eigen_dirs} ${small_vector_dirs} -std=c++23
        DEPENDS ${headers}
        USES_TERMINAL
    )
    add_custom_target(
        ${target}_docstrings
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
    )
endfunction()
