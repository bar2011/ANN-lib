# Compile main library
add_library(ANN SHARED ann/ann.cpp ann/layers/dense.cpp)

target_include_directories(ANN PUBLIC
          "${CMAKE_CURRENT_SOURCE_DIR}/ann"
          "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ANN PRIVATE MathHelpers MNistLoader)

# Add executable
add_executable(NeuralNetwork_exec main.cpp)
target_link_libraries(NeuralNetwork_exec ANN MNistLoader)
