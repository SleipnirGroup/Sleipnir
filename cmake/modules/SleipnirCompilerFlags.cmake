macro(sleipnir_compiler_flags target)
  if (NOT MSVC)
    target_compile_options(${target} PRIVATE -Wall -pedantic -Wextra -Werror -Wno-unused-parameter)

    # Disable warning false positives in Eigen
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "8")
      target_compile_options(${target} PRIVATE -Wno-class-memaccess)
    endif()
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "11")
      target_compile_options(${target} PRIVATE -Wno-maybe-uninitialized)
    endif()
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "12")
      target_compile_options(${target} PRIVATE -Wno-array-bounds)
    endif()
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND # Resolve an error with emscripten
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "3")
      target_compile_options(${target} PRIVATE -Wno-error=unused-but-set-variable)
    endif()
  else()
    target_compile_options(${target} PRIVATE /wd4146 /wd4244 /wd4251 /wd4267 /WX)
  endif()

  target_compile_features(${target} PUBLIC cxx_std_20)
  if (MSVC)
    target_compile_options(${target} PUBLIC /bigobj)
  endif()
endmacro()
