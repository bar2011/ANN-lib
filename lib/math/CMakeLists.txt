# Define the math helpers as a header-only interface library
add_library(MathHelpers INTERFACE)

# Set include directories for public headers
target_include_directories(MathHelpers INTERFACE include)

# Ensure .tpp files are not exposed to users
target_sources(MathHelpers
    PRIVATE
    include/math/matrix.tpp
    include/math/matrixView.tpp
    include/math/vector.tpp
    include/math/vectorView.tpp
    include/math/linear.tpp
    include/math/dot.tpp
    include/math/random.tpp
)

# Note: The .tpp files are included in the .h files internally, so users won't need access to them.
