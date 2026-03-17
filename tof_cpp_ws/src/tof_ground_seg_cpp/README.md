# tof_ground_seg_cpp

C++ ROS2 port of the Python node `grid_ground_node.py` from `tof_ws/src/tof_ground_seg`.

## Goal
- Keep the same processing structure and parameter names as the Python node.
- Keep code easy to read and modify.

## File structure
- `include/tof_ground_seg_cpp/grid_ground_node.hpp`: Node class and helper declarations.
- `src/grid_ground_node.cpp`: Full algorithm implementation.
- `src/grid_ground_main.cpp`: Minimal ROS2 entrypoint.

## Pipeline mapping (Python -> C++)
- `_pc2_to_xyz_numpy` -> `GridGroundNode::pc2_to_xyz`
- `_fit_plane_pca_robust` -> `GridGroundNode::fit_plane_pca_robust`
- `_make_pc2` -> `GridGroundNode::make_result_cloud`
- `make_normal_marker_array` -> `GridGroundNode::make_normal_marker_array`
- `cb_points` -> `GridGroundNode::cb_points`
- `_record_step_time` -> `GridGroundNode::record_step_time`

## Parameters
The C++ node keeps the same key parameter names as the Python node (topics, grid, ROI, timing, marker output).

## Build
```bash
cd /home/keqi/BHFRobot/tof_cpp_ws
mkdir -p src
# package already at: src/tof_ground_seg_cpp
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## Run
```bash
source /opt/ros/humble/setup.bash
source /home/keqi/BHFRobot/tof_cpp_ws/install/setup.bash
ros2 run tof_ground_seg_cpp grid_ground
```

  source /opt/ros/humble/setup.bash
  CCACHE_DISABLE=1 colcon build --symlink-install


  source /opt/ros/humble/setup.bash
  source /home/keqi/BHFRobot/tof_cpp_ws/install/setup.bash
  ros2 run tof_ground_seg_cpp grid_ground
------------------------------------------
  source /opt/ros/jazzy/setup.bash

  unset AMENT_PREFIX_PATH
  unset COLCON_PREFIX_PATH
  unset CMAKE_PREFIX_PATH
  unset PYTHONPATH
  unset ROS_DISTRO
  unset ROS_VERSION

  source /opt/ros/jazzy/setup.bash

  cd ~/tof_ws/tof_cpp_ws
  rm -rf build install log
  colcon build --symlink-install

  source install/setup.bash
  ros2 pkg list | grep tof_ground_seg_cpp
  ros2 run tof_ground_seg_cpp grid_ground

