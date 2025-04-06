# Define the math helpers as a header-only interface library
add_library(MathHelpers INTERFACE)

# Set include directories for users of this library
target_include_directories(MathHelpers INTERFACE include)
