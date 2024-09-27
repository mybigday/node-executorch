set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_AR "aarch64-linux-gnu-ar")
set(CMAKE_RANLIB "aarch64-linux-gnu-ranlib")
set(CMAKE_STRIP "aarch64-linux-gnu-strip")
set(CMAKE_LINKER "aarch64-linux-gnu-ld")

set(CMAKE_C_FLAGS "-march=armv8-a -target aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS "-march=armv8-a -target aarch64-linux-gnu")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")