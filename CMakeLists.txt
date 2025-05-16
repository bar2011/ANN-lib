cmake_minimum_required(VERSION 3.20)

project(NeuralNetworks VERSION 0.0.1)

# Compiler settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)		# Disable compiler extensions
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Ensure correct flags for clang/libc++
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang, setting flags for libc++")
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
endif()

# Main program
add_subdirectory(src)

# Helper libraries subdirectories
add_subdirectory(lib/utils)
add_subdirectory(lib/math)
add_subdirectory(lib/mnist)

# Enable experimental features (e.g. jthread)
add_compile_options(-fexperimental-library)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Applying debug configuration")
  add_compile_options(-g3 -Wpedantic -pedantic-errors -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror -fsanitize=address,undefined,leak -fno-omit-frame-pointer)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Applying release configuration")
  add_compile_options(-O2 -DNDEBUG -flto -march=native -s -Wl,-strip-all)
endif()
