add_library(Loaders STATIC src/mnist.cpp src/exception.cpp)

target_include_directories(Loaders PUBLIC include)
target_link_libraries(Loaders MathHelpers)
