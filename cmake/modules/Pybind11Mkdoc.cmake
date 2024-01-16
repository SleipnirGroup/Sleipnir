function(pybind11_mkdoc target headers)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    if(UNIX AND NOT APPLE)
        set(env_vars LLVM_DIR_PATH=/usr/lib LIBCLANG_PATH=/usr/lib/libclang.so)
    endif()

    get_target_property(target_dirs ${target} INCLUDE_DIRECTORIES)
    list(TRANSFORM target_dirs PREPEND "-I")

    get_target_property(eigen_dirs Eigen3::Eigen INCLUDE_DIRECTORIES)
    list(TRANSFORM eigen_dirs PREPEND "-I")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/jormungandr/cpp/Docstrings.hpp
        COMMAND
            ${env_vars} ${Python3_EXECUTABLE} -m pybind11_mkdoc ${headers} -o
            ${CMAKE_CURRENT_SOURCE_DIR}/jormungandr/cpp/Docstrings.hpp
            -I/usr/lib/clang/16/include ${target_dirs} ${eigen_dirs} -std=c++20
        DEPENDS ${headers}
        USES_TERMINAL
    )
    add_custom_target(
        ${target}_docstrings
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/jormungandr/cpp/Docstrings.hpp
    )
endfunction()
