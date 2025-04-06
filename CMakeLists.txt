cmake_minimum_required(VERSION 3.15)

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
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
