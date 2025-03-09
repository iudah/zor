cmake_minimum_required(VERSION 3.15)

project(Zor VERSION 2.1 DESCRIPTION "zor tensor library")

set(TARGET zor)

# Set C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Define compiler warnings in one place
set(C_WARNINGS "-Wall -Wextra -Wshadow -Wformat=2 -Wunused")

# Set flags for Debug and Release configurations
set(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -lm -mfpu=neon")
set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -static-libsan -mfpu=neon")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -O3 ")

# Set version script
set(VERSION_SCRIPT ${CMAKE_SOURCE_DIR}/zor.map.txt)
if(EXISTS ${VERSION_SCRIPT})
  set(VERSION_SCRIPT_FLAGS "-Wl,--version-script,${VERSION_SCRIPT}" "-Wl,--no-undefined-version")
else()
  set(VERSION_SCRIPT_FLAGS "")
endif()

# Include source and header files
include(sources.cmake)
include(headers.cmake)

include(ExternalProject)

ExternalProject_Add(
  Zot
  SOURCE_DIR /sdcard/Jay/Projects/xab_laud_ubject/zot
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/zot
  BUILD_ALWAYS TRUE
  BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/zot/lib/libzot.so
)

set(PCG_SRC /sdcard/Jay/Projects/xab_laud_ubject/pcg-c)

ExternalProject_Add(
  pcg-c
  SOURCE_DIR ${PCG_SRC}
  PATCH_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/pcg/lib ${CMAKE_BINARY_DIR}/pcg/include && sed "s|PREFIX = /usr/local|PREFIX = ${CMAKE_BINARY_DIR}/pcg|" Makefile | sed -E "/cd / { /cd src/! s|^|#| }" > ${PCG_SRC}/termux.patch.cmake
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -C ${PCG_SRC}/ -f ${PCG_SRC}/termux.patch.cmake
  INSTALL_COMMAND make -C ${PCG_SRC}/ -f ${PCG_SRC}/termux.patch.cmake install
  BUILD_ALWAYS TRUE
)

# Create INTERFACE target for shared flags
add_library(zor_c_flags INTERFACE)
target_compile_features(zor_c_flags INTERFACE c_std_23)
target_compile_options(
  zor_c_flags
  INTERFACE
    "$<BUILD_INTERFACE:${C_WARNINGS}>"
)
target_link_options(
  zor_c_flags
  INTERFACE
    ${VERSION_SCRIPT_FLAGS}
)


# Main library setup
add_library(${TARGET} SHARED)
target_sources(${TARGET} PRIVATE ${SOURCE_FILES})
target_sources(${TARGET} PRIVATE ${PCG_SRC}/extras/entropy.c)
target_link_options(${TARGET} PRIVATE ${VERSION_SCRIPT_FLAGS})
set_target_properties(${TARGET} PROPERTIES LINK_DEPENDS ${VERSION_SCRIPT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/zot/include )
target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/pcg/include )
target_include_directories(${TARGET} PRIVATE ${PCG_SRC}/extras )
target_link_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/zot/lib/ ${CMAKE_BINARY_DIR}/pcg/lib/)
target_link_libraries(${TARGET} PRIVATE zot pcg_random)
add_dependencies(${TARGET} Zot pcg-c)

add_executable(example-0 examples/example-0.c)
target_link_libraries(example-0 PRIVATE ${TARGET})
target_include_directories(example-0 PRIVATE ${CMAKE_BINARY_DIR}/zot/include )
target_link_libraries(example-0 PRIVATE ${CMAKE_BINARY_DIR}/zot/lib/libzot.so)

# Tests section
# Option to enable or disable tests
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
  include(include_google_test.cmake)
  add_executable(zor_test tests/zortest.cpp )
  target_link_libraries(zor_test PRIVATE ${TARGET} gtest gtest_main)
  enable_testing()

  include(GoogleTest)
  gtest_discover_tests(zor_test)
endif()

# Install targets
install(TARGETS ${TARGET} DESTINATION bin)
install(FILES ${HEADER_FILES} DESTINATION include)
