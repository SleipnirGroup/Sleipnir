function(pybind11_stubgen target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND
            ${Python3_EXECUTABLE} -m pybind11_stubgen --ignore-all-errors
            --print-invalid-expressions-as-is --exit-code
            $<TARGET_FILE_BASE_NAME:${target}> -o
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
