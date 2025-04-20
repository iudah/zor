cmake_minimum_required(VERSION 3.15)

project(Zor VERSION 2.1 DESCRIPTION "zor tensor library")

set(TARGET zor)

# Set C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Define compiler warnings in one place
set(C_WARNINGS "-Wall -Wextra -Wshadow -Wformat=2 -Wunused")

# Set flags for Debug and Release configurations
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm -mfpu=neon")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -static-libsan -mfpu=neon")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 ")

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

list(APPEND CMAKE_PREFIX_PATH
  "${CMAKE_BINARY_DIR}/memalloc/lib/cmake/memalloc"
  "${CMAKE_BINARY_DIR}/zot/lib/cmake/zot")

find_package(memalloc QUIET)
if(NOT memalloc_FOUND)
  ExternalProject_Add(
    MemAlloc
    SOURCE_DIR /sdcard/Jay/Projects/xab_laud_ubject/memalloc/
    INSTALL_DIR ${CMAKE_BINARY_DIR}/memalloc
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DBUILD_EXEC=OFF
    BUILD_ALWAYS OFF
    STAMP_DIR ${CMAKE_BINARY_DIR}/stamps
  )
endif()

find_package(zot QUIET)
if(NOT zot_FOUND)
  ExternalProject_Add(
    Zot
    SOURCE_DIR /sdcard/Jay/Projects/xab_laud_ubject/zot/
    INSTALL_DIR ${CMAKE_BINARY_DIR}/zot
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    BUILD_ALWAYS OFF
    STAMP_DIR ${CMAKE_BINARY_DIR}/stamps
  )
endif()

set(PCG_SRC /sdcard/Jay/Projects/xab_laud_ubject/pcg-c)
ExternalProject_Add(
  pcg-c
  SOURCE_DIR ${PCG_SRC}
  INSTALL_DIR ${CMAKE_BINARY_DIR}/pcg
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -C ${PCG_SRC}/src PREFIX=<INSTALL_DIR>
  INSTALL_COMMAND pwd && install -d <INSTALL_DIR>/lib/ && install -d <INSTALL_DIR>/include/ && install ${PCG_SRC}/src/libpcg_random.a <INSTALL_DIR>/lib/ && install -m 0644 ${PCG_SRC}/include/pcg_variants.h <INSTALL_DIR>/include/
  BUILD_ALWAYS OFF
  BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libpcg_random.a
)

ExternalProject_Get_Property(pcg-c INSTALL_DIR)
message(STATUS "PCG ${INSTALL_DIR}")

add_library(pcg_random STATIC IMPORTED)
set_target_properties(pcg_random PROPERTIES
  IMPORTED_LOCATION ${INSTALL_DIR}/lib/libpcg_random.a
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${PCG_SRC}/include>;$<INSTALL_INTERFACE:include>")
add_dependencies(pcg_random pcg-c)

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
target_sources(${TARGET} PRIVATE
  ${SOURCE_FILES}
  ${PCG_SRC}/extras/entropy.c)
target_link_options(${TARGET} PRIVATE ${VERSION_SCRIPT_FLAGS})
set_target_properties(${TARGET} PROPERTIES LINK_DEPENDS ${VERSION_SCRIPT})
target_include_directories(${TARGET} PRIVATE
  include
  ${CMAKE_BINARY_DIR}/pcg/include
  ${PCG_SRC}/extras)
target_link_libraries(${TARGET} PRIVATE memalloc zot pcg_random)
add_dependencies(${TARGET} pcg_random)

if(NOT memalloc_FOUND)
  target_link_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/memalloc/lib)
  add_dependencies(${TARGET} MemAlloc)
  add_dependencies(Zot MemAlloc)
endif()

if(NOT zot_FOUND)
  target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/zot/include)
  target_link_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/zot/lib)
  add_dependencies(${TARGET} Zot)
endif()

add_executable(example-0 examples/example-0.c)
target_include_directories(example-0 PRIVATE
  include
)
target_link_libraries(example-0 PRIVATE ${TARGET})
add_dependencies(example-0 ${TARGET})


# Tests section
# Option to enable or disable tests
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
  include(include_google_test.cmake)
  add_executable(zor_test tests/zortest.cpp)
  target_link_libraries(zor_test PRIVATE ${TARGET} gtest gtest_main)
  enable_testing()

  include(GoogleTest)
  gtest_discover_tests(zor_test)
endif()

# Install targets
install(TARGETS ${TARGET} EXPORT ZorTargets LIBRARY DESTINATION lib INCLUDES DESTINATION include)
install(FILES ${HEADER_FILES} DESTINATION include)
install(EXPORT ZorTargets FILE zor-config.cmake DESTINATION lib/cmake/zor)

export(EXPORT ZorTargets FILE "${CMAKE_BINARY_DIR}/zor-config.cmake")
