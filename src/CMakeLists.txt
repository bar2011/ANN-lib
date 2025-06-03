# Compile main library
add_library(
  ANN STATIC
  "ann/ann.cpp"
  "ann/feedForwardModel.cpp"
  "ann/modelLoader.cpp"
  "ann/layers/dense.cpp"
  "ann/layers/dropout.cpp"
  "ann/loss/MSE.cpp"
  "ann/loss/MAE.cpp"
  "ann/loss/binary.cpp"
  "ann/loss/categorical.cpp"
  "ann/loss/categoricalSoftmax.cpp"
  "ann/loss/loss.cpp"
  "ann/activations/step.cpp"
  "ann/activations/relu.cpp"
  "ann/activations/sigmoid.cpp"
  "ann/activations/leakyRelu.cpp"
  "ann/activations/softmax.cpp"
  "ann/optimizers/sgd.cpp"
  "ann/optimizers/adagrad.cpp"
  "ann/optimizers/rmsprop.cpp"
  "ann/optimizers/adam.cpp")

target_include_directories(ANN PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ann"
                                      "${CMAKE_SOURCE_DIR}/include")

# Make .tpp files not exposed to users
set_source_files_properties(PROPERTIES HEADER_FILE_ONLY True)

target_link_libraries(ANN PRIVATE MathHelpers Utils)

# Add executable
add_executable(NeuralNetwork_exec main.cpp)
target_link_libraries(NeuralNetwork_exec PRIVATE ANN Loaders)
