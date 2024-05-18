function(pybind11_stubgen target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND
            ${Python3_EXECUTABLE} -m pybind11_stubgen --numpy-array-use-type-var
            --ignore-unresolved-names
            'numpy.float64|numpy.ndarray|scipy.sparse.csc_matrix' --exit-code
            $<TARGET_FILE_BASE_NAME:${target}> -o
            $<TARGET_FILE_DIR:${target}>-stubs
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fix_stubgen.py
            $<TARGET_FILE_DIR:${target}>-stubs
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
        USES_TERMINAL
    )
endfunction()

function(pybind11_stubgen_install target destination)
    install(
        DIRECTORY $<TARGET_FILE_DIR:${target}>-stubs/${target}/
        COMPONENT python_modules
        DESTINATION ${destination}
        FILES_MATCHING
        PATTERN "*.pyi"
    )
endfunction()
