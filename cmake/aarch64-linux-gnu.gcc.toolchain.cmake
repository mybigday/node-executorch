set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm64)

find_program(
  AARCH64_LINUX_CC NAMES aarch64-linux-gnu-gcc-13 aarch64-linux-gnu-gcc-14
                         aarch64-linux-gnu-gcc)
find_program(
  AARCH64_LINUX_CXX NAMES aarch64-linux-gnu-g++-13 aarch64-linux-gnu-g++-14
                          aarch64-linux-gnu-g++)
if(NOT AARCH64_LINUX_CC)
  message(FATAL_ERROR "aarch64-linux-gnu-gcc not found")
endif()

set(CMAKE_C_COMPILER ${AARCH64_LINUX_CC})
set(CMAKE_CXX_COMPILER ${AARCH64_LINUX_CXX})
set(CMAKE_AR "aarch64-linux-gnu-ar")
set(CMAKE_RANLIB "aarch64-linux-gnu-ranlib")
set(CMAKE_STRIP "aarch64-linux-gnu-strip")
set(CMAKE_LINKER "aarch64-linux-gnu-ld")
