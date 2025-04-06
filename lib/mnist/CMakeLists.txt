add_library(MNistLoader STATIC src/loader.cpp src/exception.cpp)

target_include_directories(MNistLoader PUBLIC include)
target_link_libraries(MNistLoader MathHelpers)
