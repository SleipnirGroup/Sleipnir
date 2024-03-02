macro(sleipnir_compiler_flags target)
    if(NOT MSVC)
        target_compile_options(
            ${target}
            PRIVATE -Wall -pedantic -Wextra -Werror -Wno-unused-parameter
        )
    else()
        # Suppress the following warnings:
        #   * C4244: lossy conversion
        #   * C4251: missing dllexport/dllimport attribute on data member
        target_compile_options(${target} PRIVATE /wd4244 /wd4251 /WX)
    endif()

    target_compile_features(${target} PUBLIC cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PUBLIC /utf-8 /bigobj)
    endif()
endmacro()
