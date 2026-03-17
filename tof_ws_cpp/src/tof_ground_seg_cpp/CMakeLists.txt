cmake_minimum_required(VERSION 3.8)
project(tof_ground_seg_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(grid_ground_node
  src/grid_ground_node.cpp
  src/main.cpp
)

target_compile_features(grid_ground_node PUBLIC cxx_std_17)
target_include_directories(grid_ground_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(grid_ground_node
  geometry_msgs
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
)
target_link_libraries(grid_ground_node Eigen3::Eigen)

install(TARGETS grid_ground_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
