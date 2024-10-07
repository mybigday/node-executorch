set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR arm64)

set(CMAKE_C_COMPILER aarch64-w64-mingw32-clang)
set(CMAKE_CXX_COMPILER aarch64-w64-mingw32-clang++)
set(CMAKE_RC_COMPILER aarch64-w64-mingw32-windres)
set(CMAKE_AR aarch64-w64-mingw32-ar)
set(CMAKE_RANLIB aarch64-w64-mingw32-ranlib)
set(CMAKE_STRIP aarch64-w64-mingw32-strip)
set(CMAKE_LINKER aarch64-w64-mingw32-ld)

set(CMAKE_C_FLAGS "-march=armv8.4-a")
set(CMAKE_CXX_FLAGS "-march=armv8.4-a")

# cache flags
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS}"
    CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS}"
    CACHE STRING "c++ flags")
