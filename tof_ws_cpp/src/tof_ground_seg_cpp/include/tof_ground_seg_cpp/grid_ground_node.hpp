#ifndef TOF_GROUND_SEG_CPP__GRID_GROUND_NODE_HPP_
#define TOF_GROUND_SEG_CPP__GRID_GROUND_NODE_HPP_

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace tof_ground_seg_cpp
{

struct GridCellPlane
{
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  float confidence = 0.0F;
  int cell_x = 0;
  int cell_y = 0;
};

class GridGroundNode : public rclcpp::Node
{
public:
  GridGroundNode();
  ~GridGroundNode() override;

private:
  struct RuntimeConfig
  {
    std::string up_axis;
    double cell_size = 0.03;
    int min_points_per_cell = 20;
    double outlier_dist = 0.03;
    double min_range = 0.05;
    double max_range = 6.0;
    bool prefer_normal_positive_z = true;
    double d_max = 0.035;
    bool use_8_neighbors = true;
    int ground_components_keep = 1;
    bool publish_normals_markers = true;
    double normal_length = 0.10;
    bool publish_raw_debug_cloud = false;
    int raw_debug_max_points = 20000;
    bool verbose_debug_logs = false;
    std::array<double, 3> mins{{-2.0, -2.0, -2.0}};
    std::array<double, 3> maxs{{2.0, 2.0, 2.0}};
  };

  void declare_parameters();
  RuntimeConfig get_runtime_config() const;
  void record_step_time(double dt_ms);
  void handle_points(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ground_normals_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nonground_normals_pub_;

  bool timing_histogram_enabled_ = true;
  int timing_histogram_window_ = 200;
  int timing_histogram_report_every_ = 30;
  bool timing_log_each_step_ = false;
  bool timing_save_txt_ = true;
  std::string timing_txt_path_ =
    "/home/keqi/BFHRobotic/Plot/Real_time/tof_cpp_step_time_ms.txt";
  std::vector<double> step_times_ms_;
  int timing_samples_since_report_ = 0;
  std::vector<double> timing_bins_ms_;
  std::unique_ptr<std::ofstream> timing_txt_file_;
};

}  // namespace tof_ground_seg_cpp

#endif  // TOF_GROUND_SEG_CPP__GRID_GROUND_NODE_HPP_
