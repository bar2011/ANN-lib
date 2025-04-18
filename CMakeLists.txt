cmake_minimum_required(VERSION 3.20)

project(NeuralNetworks VERSION 0.0.1)

# Compiler settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)		# Disable compiler extensions
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Main program
add_subdirectory(src)

# Helper libraries subdirectories
add_subdirectory(lib/math)
add_subdirectory(lib/mnist)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Applying debug configuration")
  add_compile_options(-g3 -Wpedantic -pedantic-errors -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror -fsanitize=address,undefined,leak -fno-omit-frame-pointer)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message("Applying release configuration")
  add_compile_options(-O2 -DNDEBUG -flto -march=native -s -Wl,-strip-all)
endif()
