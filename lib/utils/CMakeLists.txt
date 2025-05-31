add_library(Utils STATIC src/parallel.cpp)

# Ensure the static library is compiled with -fPIC
set_target_properties(Utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(Utils PUBLIC include)
