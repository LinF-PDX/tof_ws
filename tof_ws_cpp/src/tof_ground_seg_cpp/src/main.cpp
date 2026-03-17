#include "rclcpp/rclcpp.hpp"

#include "tof_ground_seg_cpp/grid_ground_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<tof_ground_seg_cpp::GridGroundNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
