# ToF Ground Segmentation Run Guide

This repository has two versions of the same ground-segmentation node:

- Python workspace: `tof_ws`
- C++ workspace: `tof_ws_cpp`

The bag used here contains the point cloud on:

```bash
/my_blaze/pylon_ros2_camera_node/blaze_cloud
```

## Bag File

```bash
/home/keqi/BFHRobotic/bhf_tof_cam_imu_correct_0.db3
```

## Python Package

Build:

```bash
cd /home/keqi/BFHRobotic/tof_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Run:

```bash
cd /home/keqi/BFHRobotic/tof_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run tof_ground_seg grid_ground --ros-args \
  -p points_topic:=/my_blaze/pylon_ros2_camera_node/blaze_cloud
```

## C++ Package

Build:

```bash
cd /home/keqi/BFHRobotic/tof_ws_cpp
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Run:

```bash
cd /home/keqi/BFHRobotic/tof_ws_cpp
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run tof_ground_seg_cpp grid_ground_node --ros-args \
  -p points_topic:=/my_blaze/pylon_ros2_camera_node/blaze_cloud
```

## Play The Bag

Open another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/keqi/BFHRobotic/bhf_tof_cam_imu_correct_0.db3
```

## Echo Ground Output

Python workspace:

```bash
source /opt/ros/humble/setup.bash
source /home/keqi/BFHRobotic/tof_ws/install/setup.bash
ros2 topic echo /ground_grid_cells
```

C++ workspace:

```bash
source /opt/ros/humble/setup.bash
source /home/keqi/BFHRobotic/tof_ws_cpp/install/setup.bash
ros2 topic echo /ground_grid_cells
```

You can also check:

```bash
ros2 topic list | grep ground
ros2 node info /tof_grid_ground
```

## RViz Static Transforms

When using RViz, run these in separate terminals:

```bash
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher \
0 0 0 0 0 3.14159 base_link pylon_camera
```

```bash
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher \
-0.145 0 0.125 -1.571 0 1.571 base_link imu_link
```

## Timing Logs

Each node writes one callback time per line in milliseconds.

Python output:

```bash
/home/keqi/BFHRobotic/Plot/Real_time/tof_python_step_time_ms.txt
```

C++ output:

```bash
/home/keqi/BFHRobotic/Plot/Real_time/tof_cpp_step_time_ms.txt
```

Watch them live:

```bash
tail -f /home/keqi/BFHRobotic/Plot/Real_time/tof_python_step_time_ms.txt
tail -f /home/keqi/BFHRobotic/Plot/Real_time/tof_cpp_step_time_ms.txt
```

## Recommended Terminal Layout

Python test:

1. Terminal 1: run the Python node
2. Terminal 2: play the bag
3. Terminal 3: echo `/ground_grid_cells`
4. Terminal 4: run the static transforms for RViz

C++ test:

1. Terminal 1: run the C++ node
2. Terminal 2: play the bag
3. Terminal 3: echo `/ground_grid_cells`
4. Terminal 4: run the static transforms for RViz
