cmake_minimum_required(VERSION 3.8)
project(tof_ground_seg_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(grid_ground_node_lib
  src/grid_ground_node.cpp
)

target_include_directories(grid_ground_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(grid_ground_node_lib PUBLIC cxx_std_17)

target_link_libraries(grid_ground_node_lib Eigen3::Eigen)

ament_target_dependencies(grid_ground_node_lib
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
)

add_executable(grid_ground src/grid_ground_main.cpp)
target_link_libraries(grid_ground grid_ground_node_lib)
ament_target_dependencies(grid_ground rclcpp)

install(TARGETS
  grid_ground_node_lib
  grid_ground
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES
  resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
