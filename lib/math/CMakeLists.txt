# Define the math helpers as a header-only interface library
add_library(MathHelpers INTERFACE)

# Set include directories for public headers
target_include_directories(MathHelpers INTERFACE include)

# Ensure .tpp files are not exposed to users
set_source_files_properties(
    include/math/matrix.tpp
    include/math/matrixView.tpp
    include/math/vector.tpp
    include/math/vectorView.tpp
    include/math/linear.tpp
    include/math/random.tpp
    PROPERTIES HEADER_FILE_ONLY True
)

# Note: The .tpp files are included in the .h files internally, so users won't need access to them.
