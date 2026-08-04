function(pybind11_mkdoc target headers)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    if(UNIX AND NOT APPLE)
        if(EXISTS /usr/lib/libclang.so)
            set(env_vars
                LLVM_DIR_PATH=/usr/lib
                LIBCLANG_PATH=/usr/lib/libclang.so
            )
        else()
            # Get default clang version
            execute_process(
                COMMAND
                    bash -c
                    "clang++ --version | grep -E -o \'[0-9]+\' | head -1"
                OUTPUT_VARIABLE CLANG_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY
            )

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

    set(docstrings ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp)
    set(generated_docstrings ${CMAKE_CURRENT_BINARY_DIR}/docstrings.hpp)

    add_custom_command(
        OUTPUT ${generated_docstrings}
        COMMAND
            ${env_vars} ${Python3_EXECUTABLE} -m pybind11_mkdoc ${headers} -o
            ${generated_docstrings}
            -I/usr/lib/clang/`clang++ --version | grep -E -o '[0-9]+' | head
            -1`/include ${target_dirs} ${eigen_dirs} ${small_vector_dirs}
            -std=c++23
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${generated_docstrings} ${docstrings}
        COMMAND ${CMAKE_COMMAND} -E remove ${generated_docstrings}
        DEPENDS ${headers}
        USES_TERMINAL
    )
    add_custom_target(${target}_docstrings DEPENDS ${generated_docstrings})
endfunction()
