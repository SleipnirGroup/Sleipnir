function(pybind11_mkdoc target headers)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Check that pybind11_mkdoc is installed
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11_mkdoc"
        RESULT_VARIABLE pybind11_mkdoc_result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT pybind11_mkdoc_result EQUAL 0)
        message(
            FATAL_ERROR
            "pybind11_mkdoc not found for ${Python3_EXECUTABLE}. Install it with `pip install --group docstrings`."
        )
    endif()

    # Target compiler flags (evaluated at build time since the include
    # directories may contain generator expressions)
    set(target_flags
        "-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    )

    if(UNIX AND NOT APPLE)
        # Get default clang version
        execute_process(
            COMMAND clang++ --version
            OUTPUT_VARIABLE clang_version
            COMMAND_ERROR_IS_FATAL ANY
        )
        string(REGEX MATCH "[0-9]+" clang_version ${clang_version})

        # LLVM environment variables
        if(EXISTS /usr/lib/libclang.so)
            set(env_vars
                LLVM_DIR_PATH=/usr/lib
                LIBCLANG_PATH=/usr/lib/libclang.so
            )
        elseif(EXISTS /usr/lib/llvm-${clang_version}/lib/libclang.so)
            set(env_vars
                LLVM_DIR_PATH=/usr/lib/llvm-${clang_version}
                LIBCLANG_PATH=/usr/lib/llvm-${clang_version}/lib/libclang.so
            )
        endif()

        # Clang compiler flags
        set(clang_flags -I/usr/lib/clang/${clang_version}/include)
    endif()

    # clang on Windows aarch64 failed to parse Eigen's NEON headers
    list(APPEND clang_flags -DEIGEN_DONT_VECTORIZE)

    # TODO: Remove when Python 3.15 makes UTF-8 the default
    list(APPEND env_vars PYTHONUTF8=1)

    # Generate docstrings.hpp
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
        COMMAND
            ${CMAKE_COMMAND} -E env ${env_vars} ${Python3_EXECUTABLE} -m
            pybind11_mkdoc ${headers} -o
            ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
            ${target_flags} ${clang_flags} -std=c++23
        COMMAND
            ${CMAKE_COMMAND}
            -DFILE=${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp -P
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Pybind11MkdocPostprocess.cmake
        DEPENDS
            ${headers}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Pybind11MkdocPostprocess.cmake
        COMMAND_EXPAND_LISTS
    )

    # Add docstrings target
    add_custom_target(
        ${target}_docstrings
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
    )
endfunction()
