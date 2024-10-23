cmake_minimum_required(VERSION 3.10)
project(BsplineSE3Example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

set(OPENVINS_INCLUDE_DIR /usr/local/include/open_vins/)
set(OPENVINS_LIBRARIES ov_msckf_lib)

# Executable
add_executable(bspline_example bspline_example.cpp)

# Include directories
target_include_directories(bspline_example PRIVATE ${OPENVINS_INCLUDE_DIRS} ${Pangolin_LIBRARIES})
# Link libraries
target_link_libraries(bspline_example PRIVATE Eigen3::Eigen ${OPENVINS_LIBRARIES} ${Pangolin_INCLUDE_DIRS})
