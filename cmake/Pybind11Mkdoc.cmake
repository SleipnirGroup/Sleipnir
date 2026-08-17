function(pybind11_mkdoc target headers)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    get_target_property(target_dirs ${target} INCLUDE_DIRECTORIES)
    list(TRANSFORM target_dirs PREPEND "-I")

    get_target_property(eigen_dirs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    list(FILTER eigen_dirs INCLUDE REGEX "\\$<BUILD_INTERFACE:.*>")
    list(TRANSFORM eigen_dirs PREPEND "-I")

    # Get default clang version
    execute_process(
        COMMAND clang++ --version
        OUTPUT_VARIABLE clang_version
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(REGEX MATCH "[0-9]+" clang_version ${clang_version})

    if(UNIX AND NOT APPLE)
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
        set(llvm_dirs -I/usr/lib/clang/${clang_version}/include)
    endif()

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
            ${CMAKE_COMMAND} -E env ${env_vars} ${Python3_EXECUTABLE} -m
            pybind11_mkdoc ${headers} -o
            ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp ${target_dirs}
            ${eigen_dirs} ${llvm_dirs} ${small_vector_dirs} -std=c++23
        DEPENDS ${headers}
    )
    add_custom_target(
        ${target}_docstrings
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/python/cpp/docstrings.hpp
    )
endfunction()
