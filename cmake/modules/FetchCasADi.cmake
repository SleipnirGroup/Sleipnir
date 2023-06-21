macro(fetch_casadi)
  cmake_POLICY(SET CMP0135 NEW)
  set(CASADI_LIBDIR ${CMAKE_BINARY_DIR}/_deps/casadi-src/casadi)
  set(CASADI_INCLUDEDIR ${CMAKE_BINARY_DIR}/_deps/casadi-src/casadi/include)
  if (${CMAKE_SYSTEM_NAME} MATCHES "MINGW" OR ${CMAKE_SYSTEM_NAME} MATCHES "MSYS" OR WIN32)
    message(STATUS "Building for Windows")
    set(CASADI_URL https://github.com/casadi/casadi/releases/download/3.6.3/casadi-3.6.3-windows64-py311.zip)
    set(CASADI_INSTALL_LIBS
      ${CASADI_LIBDIR}/libcasadi.dll
      ${CASADI_LIBDIR}/libstdc++-6.dll
      ${CASADI_LIBDIR}/libcasadi_nlpsol_ipopt.dll
      ${CASADI_LIBDIR}/libquadmath-0.dll
      ${CASADI_LIBDIR}/libgcc_s_seh-1.dll)
    set(CASADI_INSTALL_DEST "bin")
  elseif (APPLE)
    if (CMAKE_APPLE_SILICON_PROCESSOR MATCHES "arm64")
      message(STATUS "Building for macOS arm64")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};@loader_path/../lib;@loader_path")
      set(CASADI_URL https://github.com/casadi/casadi/releases/download/3.6.3/casadi-3.6.3-osx_arm64-py311.zip)
      set(CASADI_INSTALL_LIBS
        ${CASADI_LIBDIR}/libcasadi.3.7.dylib
        ${CASADI_LIBDIR}/libc++.1.0.dylib
        ${CASADI_LIBDIR}/libcasadi_nlpsol_ipopt.dylib
        ${CASADI_LIBDIR}/libipopt.3.dylib
        ${CASADI_LIBDIR}/libcoinmumps.3.dylib
        ${CASADI_LIBDIR}/libcoinmetis.2.dylib
        ${CASADI_LIBDIR}/libgfortran.5.dylib
        ${CASADI_LIBDIR}/libquadmath.0.dylib
        ${CASADI_LIBDIR}/libgcc_s.1.1.dylib)
    elseif(CMAKE_APPLE_SILICON_PROCESSOR MATCHES "x86_64")
      message(STATUS "Building for macOS x86_64")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};@loader_path/../lib;@loader_path")
      set(CASADI_URL https://github.com/casadi/casadi/releases/download/3.6.3/casadi-3.6.3-osx64-py311.zip)
      set(CASADI_INSTALL_LIBS
        ${CASADI_LIBDIR}/libcasadi.3.7.dylib
        ${CASADI_LIBDIR}/libc++.1.0.dylib
        ${CASADI_LIBDIR}/libcasadi_nlpsol_ipopt.3.7.dylib
        ${CASADI_LIBDIR}/libipopt.3.dylib
        ${CASADI_LIBDIR}/libcoinmumps.3.dylib
        ${CASADI_LIBDIR}/libcoinmetis.2.dylib
        ${CASADI_LIBDIR}/libgfortran.5.dylib
        ${CASADI_LIBDIR}/libquadmath.0.dylib
        ${CASADI_LIBDIR}/libgcc_s.1.dylib
        ${CASADI_LIBDIR}/libgcc_s.1.1.dylib)
    endif()
    set(CASADI_INSTALL_DEST "lib")
  elseif (UNIX)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM64")
      message(STATUS "Building for Linux arm64")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../lib;$ORIGIN")
      set(CASADI_URL https://github.com/casadi/casadi/releases/download/3.6.3/casadi-3.6.3-linux-aarch64-py311.zip)
      set(CASADI_INSTALL_LIBS
        ${CASADI_LIBDIR}/libcasadi.so
        ${CASADI_LIBDIR}/libcasadi_nlpsol_ipopt.so)
    else()
      message(STATUS "Building for Linux x64")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../lib;$ORIGIN")
      set(CASADI_URL https://github.com/casadi/casadi/releases/download/3.6.3/casadi-3.6.3-linux64-py311.zip)
      set(CASADI_INSTALL_LIBS
        ${CASADI_LIBDIR}/libcasadi.so
        ${CASADI_LIBDIR}/libcasadi_nlpsol_ipopt.so)
    endif()
    set(CASADI_INSTALL_DEST "lib")
  endif()
  message(STATUS "Downloading CasADi from ${CASADI_URL}")

  include(FetchContent)

  FetchContent_Declare(
    casadi
    URL ${CASADI_URL}
  )

  FetchContent_MakeAvailable(casadi)
endmacro()
