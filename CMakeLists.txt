cmake_minimum_required(VERSION 3.20)

project(NeuralNetworks VERSION 0.0.1)

# Compiler settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Disable compiler extensions

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization settings

# How many iterations of a loop would require parallelization for the simplest
# worker (e.g. single addition in each loop)
set(PARALLEL_COST_MINIMUM
    10000
    CACHE STRING "Minimum estimated work units to parallelize")
add_compile_definitions(PARALLEL_COST_MINIMUM=${PARALLEL_COST_MINIMUM})

# Configure build if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE
        STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

# Compiler specific settings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using Clang compiler")

  add_compile_options(-stdlib=libc++)
  add_link_options(-stdlib=libc++)

  # Enable experimental features
  add_compile_options(-fexperimental-library)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GNU GCC compiler")
elseif(MSVC)
  message(STATUS "Using MSVC compiler")
else()
  message(WARNING "Unknown compiler. Using default flags only")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Applying debug configuration")

  if(MSVC)
    add_compile_options(/Zi /W4 /permissive- /WX /EHsc /std:c++23)
  else()
    add_compile_options(
      -g3
      -Wpedantic
      -pedantic-errors
      -Wall
      -Weffc++
      -Wextra
      -Wconversion
      -Wsign-conversion
      -Werror
      -fsanitize=leak
      -fno-omit-frame-pointer
      -std=c++23)
  endif()

elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Applying release configuration")
  if(MSVC)
    add_compile_options(/O2 /DNDEBUG /GL /std:c++23)
  else()
    add_compile_options(-O2 -DNDEBUG -flto -march=native -std=c++23)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # These only work on Linux/GNU ld
      add_link_options(-s -Wl,--strip-all)
    endif()
  endif()
endif()

# Helper libraries subdirectories
add_subdirectory(lib/utils)
add_subdirectory(lib/math)
add_subdirectory(lib/loaders)

# Main program
add_subdirectory(src)
