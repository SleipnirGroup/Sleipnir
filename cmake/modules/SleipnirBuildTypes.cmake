macro(
    sleipnir_add_build_type
    allowedBuildTypes
    buildTypeName
    cFlags
    cxxFlags
    exeLinkerFlags
    sharedLinkerFlags
)
    get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

    if(isMultiConfig)
        if(NOT ${buildTypeName} IN_LIST CMAKE_CONFIGURATION_TYPES)
            list(APPEND CMAKE_CONFIGURATION_TYPES ${buildTypeName})
        endif()
    else()
        list(APPEND allowedBuildTypes ${buildTypeName})
    endif()

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
endmacro()

macro(sleipnir_check_build_type allowedBuildTypes)
    get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

    if(NOT isMultiConfig)
        set_property(
            CACHE CMAKE_BUILD_TYPE
            PROPERTY STRINGS "${allowedBuildTypes}"
        )

        if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
            message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
        endif()
    endif()
endmacro()

set(allowedBuildTypes Debug Release RelWithDebInfo MinSizeRel)

if(NOT MSVC)
    sleipnir_add_build_type(
      allowedBuildTypes
      "Asan"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=address -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=address -fno-omit-frame-pointer"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address"
    )

    sleipnir_add_build_type(
      allowedBuildTypes
      "Tsan"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread"
    )

    sleipnir_add_build_type(
      allowedBuildTypes
      "Ubsan"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=undefined -fno-sanitize-recover=all"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=undefined"
    )

    sleipnir_add_build_type(
      allowedBuildTypes
      "Perf"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}"
    )

    sleipnir_add_build_type(
      allowedBuildTypes
      "Coverage"
      "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fprofile-instr-generate -fcoverage-mapping"
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fprofile-instr-generate -fcoverage-mapping"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}"
      "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}"
    )

    sleipnir_check_build_type(allowedBuildTypes)
endif()
