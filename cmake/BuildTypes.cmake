function(
    add_build_type
    buildTypeName
    cFlags
    cxxFlags
    exeLinkerFlags
    sharedLinkerFlags
)
    set_property(GLOBAL APPEND PROPERTY SLEIPNIR_BUILD_TYPES ${buildTypeName})

    string(TOUPPER ${buildTypeName} buildTypeNameUpper)

    set(CMAKE_C_FLAGS_${buildTypeNameUpper}
        ${cFlags}
        CACHE STRING
        "Flags used by the C compiler for ${buildTypeName} build type or configuration."
        FORCE
    )
    set(CMAKE_CXX_FLAGS_${buildTypeNameUpper}
        ${cxxFlags}
        CACHE STRING
        "Flags used by the C++ compiler for ${buildTypeName} build type or configuration."
        FORCE
    )
    set(CMAKE_EXE_LINKER_FLAGS_${buildTypeNameUpper}
        ${exeLinkerFlags}
        CACHE STRING
        "Linker flags to be used to create executables for ${buildTypeName} build type."
        FORCE
    )
    set(CMAKE_SHARED_LINKER_FLAGS_${buildTypeNameUpper}
        ${sharedLinkerFlags}
        CACHE STRING
        "Linker lags to be used to create shared libraries for ${buildTypeName} build type."
        FORCE
    )
endfunction()

if(NOT MSVC)
    add_build_type(
      "Asan"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=address -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=address -fno-omit-frame-pointer"
      "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address"
    )

    add_build_type(
      "Tsan"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
      "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread"
    )

    add_build_type(
      "Ubsan"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
      "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=undefined"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=undefined"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Source-based coverage for llvm-cov
        set(coverageCompileFlags "-fprofile-instr-generate -fcoverage-mapping")
        set(coverageLinkFlags "")
    else()
        # gcov-based coverage
        set(coverageCompileFlags "--coverage")
        set(coverageLinkFlags "--coverage")
    endif()
    add_build_type(
      "Coverage"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${coverageCompileFlags}"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${coverageCompileFlags}"
      "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} ${coverageLinkFlags}"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} ${coverageLinkFlags}"
    )

    block(PROPAGATE CMAKE_CONFIGURATION_TYPES)
        get_property(customBuildTypes GLOBAL PROPERTY SLEIPNIR_BUILD_TYPES)
        get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

        if(isMultiConfig)
            foreach(buildType IN LISTS customBuildTypes)
                if(NOT buildType IN_LIST CMAKE_CONFIGURATION_TYPES)
                    list(APPEND CMAKE_CONFIGURATION_TYPES ${buildType})
                endif()
            endforeach()
        else()
            set(allowedBuildTypes
                Debug
                Release
                RelWithDebInfo
                MinSizeRel
                ${customBuildTypes}
            )
            set_property(
                CACHE CMAKE_BUILD_TYPE
                PROPERTY STRINGS "${allowedBuildTypes}"
            )

            if(
                CMAKE_BUILD_TYPE
                AND NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes
            )
                message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
            endif()
        endif()
    endblock()
endif()
