function(pybind11_stubgen target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND
            ${Python3_EXECUTABLE} -m pybind11_stubgen
            $<TARGET_FILE_BASE_NAME:${target}> -o $<TARGET_FILE_DIR:${target}>
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
        USES_TERMINAL
    )
endfunction()

function(pybind11_stubgen_install target destination)
    install(
        DIRECTORY
            $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>-stubs/
        COMPONENT python_modules
        DESTINATION ${destination}/$<TARGET_FILE_BASE_NAME:${target}>
        FILES_MATCHING
        REGEX "\.pyi$"
    )
endfunction()
