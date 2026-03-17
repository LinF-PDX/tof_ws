#pragma once

#include <array>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace tof_ground_seg_cpp
{

class GridGroundNode : public rclcpp::Node
{
public:
  GridGroundNode();
  ~GridGroundNode() override;

private:
  struct CellOutput
  {
    Eigen::Vector3f center;
    Eigen::Vector3f normal;
    float confidence;
    int cx;
    int cy;
  };

  void cb_points(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void record_step_time(double dt_ms);

  std::vector<Eigen::Vector3f> pc2_to_xyz(const sensor_msgs::msg::PointCloud2 & msg) const;
  sensor_msgs::msg::PointCloud2 make_xyz_cloud(
    const std_msgs::msg::Header & header,
    const std::vector<Eigen::Vector3f> & points,
    size_t max_points) const;
  sensor_msgs::msg::PointCloud2 make_result_cloud(
    const std_msgs::msg::Header & header,
    const std::vector<CellOutput> & cells) const;
  visualization_msgs::msg::MarkerArray make_normal_marker_array(
    const std_msgs::msg::Header & header,
    const std::vector<CellOutput> & cells,
    float length,
    const std::string & ns,
    const std::array<float, 4> & rgba) const;

  bool fit_plane_pca_robust(
    const std::vector<Eigen::Vector3f> & points,
    float outlier_dist,
    Eigen::Vector3f & centroid,
    Eigen::Vector3f & normal,
    float & inlier_ratio) const;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_nonground_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_rawdebug_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_ground_normals_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_nonground_normals_;

  std::string points_topic_;
  std::string ground_topic_;
  std::string nonground_topic_;

  bool timing_histogram_enabled_;
  int timing_histogram_window_;
  int timing_histogram_report_every_;
  bool timing_log_each_step_;
  bool timing_save_txt_;
  std::string timing_txt_path_;

  std::vector<double> step_times_ms_;
  int timing_samples_since_report_;
  std::vector<double> timing_bins_ms_;
  std::ofstream timing_txt_file_;
};

}  // namespace tof_ground_seg_cpp
