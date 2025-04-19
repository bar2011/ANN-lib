# Compile main library
add_library(ANN SHARED ann/ann.cpp ann/activation.cpp)

target_include_directories(ANN PUBLIC
          "${CMAKE_CURRENT_SOURCE_DIR}/ann"
          "${CMAKE_SOURCE_DIR}/include"
)

# Make .tpp files not exposed to users
set_source_files_properties(
  "${CMAKE_SOURCE_DIR}/include/ann/layers/dense.tpp"
  PROPERTIES HEADER_FILE_ONLY True
)

target_link_libraries(ANN PRIVATE MathHelpers)

# Add executable
add_executable(NeuralNetwork_exec main.cpp)
target_link_libraries(NeuralNetwork_exec ANN MNistLoader)
