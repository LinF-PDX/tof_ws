#include "tof_ground_seg_cpp/grid_ground_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <Eigen/Eigenvalues>

#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace tof_ground_seg_cpp
{
namespace
{

class UnionFind
{
public:
  explicit UnionFind(size_t n)
  : parent_(n), size_(n, 1)
  {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(int a)
  {
    while (parent_[a] != a) {
      parent_[a] = parent_[parent_[a]];
      a = parent_[a];
    }
    return a;
  }

  void unite(int a, int b)
  {
    int ra = find(a);
    int rb = find(b);
    if (ra == rb) {
      return;
    }
    if (size_[ra] < size_[rb]) {
      std::swap(ra, rb);
    }
    parent_[rb] = ra;
    size_[ra] += size_[rb];
  }

private:
  std::vector<int> parent_;
  std::vector<int> size_;
};

struct CellCoord
{
  int x;
  int y;
  bool operator==(const CellCoord & other) const {return x == other.x && y == other.y;}
};

struct CellCoordHash
{
  std::size_t operator()(const CellCoord & k) const noexcept
  {
    return (static_cast<uint64_t>(static_cast<uint32_t>(k.x)) << 32U) ^
           static_cast<uint32_t>(k.y);
  }
};

float read_f32(const uint8_t * p, bool big_endian)
{
  uint32_t u = 0;
  if (big_endian) {
    u = (static_cast<uint32_t>(p[0]) << 24U) |
      (static_cast<uint32_t>(p[1]) << 16U) |
      (static_cast<uint32_t>(p[2]) << 8U) |
      static_cast<uint32_t>(p[3]);
  } else {
    u = (static_cast<uint32_t>(p[3]) << 24U) |
      (static_cast<uint32_t>(p[2]) << 16U) |
      (static_cast<uint32_t>(p[1]) << 8U) |
      static_cast<uint32_t>(p[0]);
  }
  float v = 0.0F;
  std::memcpy(&v, &u, sizeof(float));
  return v;
}

double percentile(const std::vector<double> & vals, double p)
{
  if (vals.empty()) {
    return 0.0;
  }
  std::vector<double> sorted = vals;
  std::sort(sorted.begin(), sorted.end());
  const double pos = (p / 100.0) * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(pos));
  const auto hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return sorted[lo];
  }
  const double t = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

int axis_index(const std::string & axis)
{
  if (axis == "x") {
    return 0;
  }
  if (axis == "y") {
    return 1;
  }
  return 2;
}

}  // namespace

GridGroundNode::GridGroundNode()
: Node("tof_grid_ground"),
  timing_samples_since_report_(0)
{
  pub_rawdebug_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/raw_points_xyz", 10);

  this->declare_parameter<std::string>("points_topic", "/camera/depth/points");
  this->declare_parameter<std::string>("ground_topic", "/ground_grid_cells");
  this->declare_parameter<std::string>("nonground_topic", "/nonground_grid_cells");

  this->declare_parameter<double>("cell_size", 0.03);
  this->declare_parameter<int>("min_points_per_cell", 20);
  this->declare_parameter<double>("outlier_dist", 0.03);
  this->declare_parameter<double>("max_range", 6.0);
  this->declare_parameter<double>("min_range", 0.05);
  this->declare_parameter<std::string>("up_axis", "z");
  this->declare_parameter<bool>("prefer_normal_positive_z", true);
  this->declare_parameter<double>("d_max", 0.035);
  this->declare_parameter<bool>("use_8_neighbors", true);
  this->declare_parameter<int>("ground_components_keep", 1);

  this->declare_parameter<double>("x_min", -2.0);
  this->declare_parameter<double>("x_max", 2.0);
  this->declare_parameter<double>("y_min", -2.0);
  this->declare_parameter<double>("y_max", 2.0);
  this->declare_parameter<double>("z_min", -2.0);
  this->declare_parameter<double>("z_max", 2.0);

  this->declare_parameter<bool>("publish_normals_markers", true);
  this->declare_parameter<double>("normal_length", 0.10);
  this->declare_parameter<std::string>("ground_normals_topic", "/ground_normals");
  this->declare_parameter<std::string>("nonground_normals_topic", "/nonground_normals");

  this->declare_parameter<bool>("timing_histogram_enabled", true);
  this->declare_parameter<int>("timing_histogram_window", 200);
  this->declare_parameter<int>("timing_histogram_report_every", 30);
  this->declare_parameter<bool>("timing_log_each_step", false);
  this->declare_parameter<bool>("timing_save_txt", true);
  this->declare_parameter<std::string>("timing_txt_path", "/tmp/tof_step_time_ms.txt");

  this->declare_parameter<bool>("publish_raw_debug_cloud", false);
  this->declare_parameter<int>("raw_debug_max_points", 20000);
  this->declare_parameter<bool>("verbose_debug_logs", false);

  points_topic_ = this->get_parameter("points_topic").as_string();
  ground_topic_ = this->get_parameter("ground_topic").as_string();
  nonground_topic_ = this->get_parameter("nonground_topic").as_string();

  pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(ground_topic_, 10);
  pub_nonground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(nonground_topic_, 10);
  pub_ground_normals_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    this->get_parameter("ground_normals_topic").as_string(), 10);
  pub_nonground_normals_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    this->get_parameter("nonground_normals_topic").as_string(), 10);

  sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    points_topic_, 10,
    std::bind(&GridGroundNode::cb_points, this, std::placeholders::_1));

  timing_histogram_enabled_ = this->get_parameter("timing_histogram_enabled").as_bool();
  timing_histogram_window_ =
    std::max(1, static_cast<int>(this->get_parameter("timing_histogram_window").as_int()));
  timing_histogram_report_every_ =
    std::max(1, static_cast<int>(this->get_parameter("timing_histogram_report_every").as_int()));
  timing_log_each_step_ = this->get_parameter("timing_log_each_step").as_bool();
  timing_save_txt_ = this->get_parameter("timing_save_txt").as_bool();
  timing_txt_path_ = this->get_parameter("timing_txt_path").as_string();

  timing_bins_ms_ = {0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 33.3, 50.0, 100.0, 200.0, 1e9};

  if (timing_save_txt_ && !timing_txt_path_.empty()) {
    std::filesystem::path path(timing_txt_path_);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    timing_txt_file_.open(timing_txt_path_, std::ios::app);
  }

  RCLCPP_INFO(this->get_logger(), "Listening: %s", points_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing ground grid cells: %s", ground_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing non-ground grid cells: %s", nonground_topic_.c_str());
  RCLCPP_INFO(
    this->get_logger(), "Timing histogram enabled=%s window=%d report_every=%d",
    timing_histogram_enabled_ ? "true" : "false",
    timing_histogram_window_, timing_histogram_report_every_);
  RCLCPP_INFO(
    this->get_logger(), "Timing per-step sample logging enabled=%s",
    timing_log_each_step_ ? "true" : "false");
  RCLCPP_INFO(
    this->get_logger(), "Timing txt enabled=%s path=%s",
    timing_save_txt_ ? "true" : "false", timing_txt_path_.c_str());
}

GridGroundNode::~GridGroundNode()
{
  if (timing_txt_file_.is_open()) {
    timing_txt_file_.close();
  }
}

std::vector<Eigen::Vector3f> GridGroundNode::pc2_to_xyz(const sensor_msgs::msg::PointCloud2 & msg) const
{
  const size_t n_msg_pts = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
  if (n_msg_pts == 0 || msg.point_step == 0) {
    return {};
  }

  int off_x = -1;
  int off_y = -1;
  int off_z = -1;
  for (const auto & f : msg.fields) {
    if (f.name == "x") {
      off_x = static_cast<int>(f.offset);
    } else if (f.name == "y") {
      off_y = static_cast<int>(f.offset);
    } else if (f.name == "z") {
      off_z = static_cast<int>(f.offset);
    }
  }
  if (off_x < 0 || off_y < 0 || off_z < 0) {
    RCLCPP_ERROR(this->get_logger(), "PointCloud2 missing x/y/z fields");
    return {};
  }

  const size_t point_step = msg.point_step;
  const size_t available_pts = msg.data.size() / point_step;
  const size_t n_pts = std::min(n_msg_pts, available_pts);
  if (n_pts == 0) {
    return {};
  }

  auto score = [&](bool decode_big_endian) {
      size_t valid = 0;
      size_t reasonable = 0;
      const size_t sample_n = std::min<size_t>(5000, n_pts);
      for (size_t i = 0; i < sample_n; ++i) {
        const uint8_t * base = &msg.data[i * point_step];
        const float x = read_f32(base + off_x, decode_big_endian);
        const float y = read_f32(base + off_y, decode_big_endian);
        const float z = read_f32(base + off_z, decode_big_endian);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }
        ++valid;
        if (std::fabs(x) < 100.0F && std::fabs(y) < 100.0F && std::fabs(z) < 100.0F) {
          ++reasonable;
        }
      }
      if (valid == 0) {
        return 0.0;
      }
      return static_cast<double>(reasonable) / static_cast<double>(valid);
    };

  const double s_le = score(false);
  const double s_be = score(true);
  const bool decode_big_endian = (s_be > s_le);

  std::vector<Eigen::Vector3f> pts;
  pts.reserve(n_pts);
  for (size_t i = 0; i < n_pts; ++i) {
    const uint8_t * base = &msg.data[i * point_step];
    const float x = read_f32(base + off_x, decode_big_endian);
    const float y = read_f32(base + off_y, decode_big_endian);
    const float z = read_f32(base + off_z, decode_big_endian);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    if (std::fabs(x) >= 1e3F || std::fabs(y) >= 1e3F || std::fabs(z) >= 1e3F) {
      continue;
    }
    pts.emplace_back(x, y, z);
  }
  return pts;
}

sensor_msgs::msg::PointCloud2 GridGroundNode::make_xyz_cloud(
  const std_msgs::msg::Header & header,
  const std::vector<Eigen::Vector3f> & points,
  size_t max_points) const
{
  sensor_msgs::msg::PointCloud2 out;
  out.header = header;
  out.height = 1;
  out.is_bigendian = false;
  out.is_dense = true;

  const size_t n = std::min(points.size(), max_points);
  out.width = static_cast<uint32_t>(n);
  out.point_step = 12;
  out.row_step = out.point_step * out.width;

  out.fields.resize(3);
  out.fields[0].name = "x";
  out.fields[0].offset = 0;
  out.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[0].count = 1;

  out.fields[1].name = "y";
  out.fields[1].offset = 4;
  out.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[1].count = 1;

  out.fields[2].name = "z";
  out.fields[2].offset = 8;
  out.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[2].count = 1;

  out.data.resize(n * out.point_step);
  for (size_t i = 0; i < n; ++i) {
    uint8_t * dst = &out.data[i * out.point_step];
    const float x = points[i].x();
    const float y = points[i].y();
    const float z = points[i].z();
    std::memcpy(dst + 0, &x, sizeof(float));
    std::memcpy(dst + 4, &y, sizeof(float));
    std::memcpy(dst + 8, &z, sizeof(float));
  }

  return out;
}

sensor_msgs::msg::PointCloud2 GridGroundNode::make_result_cloud(
  const std_msgs::msg::Header & header,
  const std::vector<CellOutput> & cells) const
{
  sensor_msgs::msg::PointCloud2 out;
  out.header = header;
  out.height = 1;
  out.width = static_cast<uint32_t>(cells.size());
  out.is_bigendian = false;
  out.is_dense = true;
  out.point_step = 28;
  out.row_step = out.point_step * out.width;

  out.fields.resize(7);
  out.fields[0].name = "x";
  out.fields[0].offset = 0;
  out.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[0].count = 1;

  out.fields[1].name = "y";
  out.fields[1].offset = 4;
  out.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[1].count = 1;

  out.fields[2].name = "z";
  out.fields[2].offset = 8;
  out.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[2].count = 1;

  out.fields[3].name = "nx";
  out.fields[3].offset = 12;
  out.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[3].count = 1;

  out.fields[4].name = "ny";
  out.fields[4].offset = 16;
  out.fields[4].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[4].count = 1;

  out.fields[5].name = "nz";
  out.fields[5].offset = 20;
  out.fields[5].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[5].count = 1;

  out.fields[6].name = "confidence";
  out.fields[6].offset = 24;
  out.fields[6].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out.fields[6].count = 1;

  out.data.resize(cells.size() * out.point_step);

  for (size_t i = 0; i < cells.size(); ++i) {
    uint8_t * dst = &out.data[i * out.point_step];
    const float vals[7] = {
      cells[i].center.x(), cells[i].center.y(), cells[i].center.z(),
      cells[i].normal.x(), cells[i].normal.y(), cells[i].normal.z(),
      cells[i].confidence};
    std::memcpy(dst, vals, sizeof(vals));
  }

  return out;
}

visualization_msgs::msg::MarkerArray GridGroundNode::make_normal_marker_array(
  const std_msgs::msg::Header & header,
  const std::vector<CellOutput> & cells,
  float length,
  const std::string & ns,
  const std::array<float, 4> & rgba) const
{
  visualization_msgs::msg::MarkerArray ma;
  ma.markers.reserve(cells.size());

  for (size_t i = 0; i < cells.size(); ++i) {
    visualization_msgs::msg::Marker m;
    m.header = header;
    m.ns = ns;
    m.id = static_cast<int>(i);
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p0;
    p0.x = cells[i].center.x();
    p0.y = cells[i].center.y();
    p0.z = cells[i].center.z();

    geometry_msgs::msg::Point p1;
    p1.x = cells[i].center.x() + length * cells[i].normal.x();
    p1.y = cells[i].center.y() + length * cells[i].normal.y();
    p1.z = cells[i].center.z() + length * cells[i].normal.z();

    m.points = {p0, p1};
    m.scale.x = 0.01;
    m.scale.y = 0.02;
    m.scale.z = 0.03;

    m.color.r = rgba[0];
    m.color.g = rgba[1];
    m.color.b = rgba[2];
    m.color.a = rgba[3];

    ma.markers.push_back(std::move(m));
  }

  return ma;
}

bool GridGroundNode::fit_plane_pca_robust(
  const std::vector<Eigen::Vector3f> & points,
  float outlier_dist,
  Eigen::Vector3f & centroid,
  Eigen::Vector3f & normal,
  float & inlier_ratio) const
{
  if (points.size() < 3) {
    return false;
  }

  auto pca_fit = [](const std::vector<Eigen::Vector3f> & pts,
      Eigen::Vector3f & out_c,
      Eigen::Vector3f & out_n)
    {
      out_c = Eigen::Vector3f::Zero();
      for (const auto & p : pts) {
        out_c += p;
      }
      out_c /= static_cast<float>(pts.size());

      Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
      for (const auto & p : pts) {
        const Eigen::Vector3f d = p - out_c;
        cov += d * d.transpose();
      }
      cov /= static_cast<float>(std::max<size_t>(1, pts.size() - 1));

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
      out_n = es.eigenvectors().col(0);
      const float n_norm = out_n.norm();
      if (n_norm < 1e-9F) {
        out_n = Eigen::Vector3f(0.0F, 0.0F, 1.0F);
      } else {
        out_n /= n_norm;
      }
    };

  Eigen::Vector3f c0;
  Eigen::Vector3f n0;
  pca_fit(points, c0, n0);

  std::vector<size_t> inliers;
  inliers.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const float d = std::fabs((points[i] - c0).dot(n0));
    if (d <= outlier_dist) {
      inliers.push_back(i);
    }
  }

  inlier_ratio = static_cast<float>(inliers.size()) / static_cast<float>(points.size());

  if (inliers.size() >= 3 && inlier_ratio < 0.999F) {
    std::vector<Eigen::Vector3f> refit_pts;
    refit_pts.reserve(inliers.size());
    for (const auto idx : inliers) {
      refit_pts.push_back(points[idx]);
    }

    Eigen::Vector3f c1;
    Eigen::Vector3f n1;
    pca_fit(refit_pts, c1, n1);

    size_t n2 = 0;
    for (const auto & p : points) {
      const float d = std::fabs((p - c1).dot(n1));
      if (d <= outlier_dist) {
        ++n2;
      }
    }
    centroid = c1;
    normal = n1;
    inlier_ratio = static_cast<float>(n2) / static_cast<float>(points.size());
    return true;
  }

  centroid = c0;
  normal = n0;
  return true;
}

void GridGroundNode::record_step_time(double dt_ms)
{
  if (timing_log_each_step_) {
    RCLCPP_INFO(this->get_logger(), "step_time_ms_sample=%.3f", dt_ms);
  }

  if (timing_txt_file_.is_open()) {
    timing_txt_file_ << std::fixed << std::setprecision(6) << dt_ms << "\n";
    timing_txt_file_.flush();
  }

  if (!timing_histogram_enabled_) {
    return;
  }

  step_times_ms_.push_back(dt_ms);
  if (static_cast<int>(step_times_ms_.size()) > timing_histogram_window_) {
    step_times_ms_.erase(
      step_times_ms_.begin(),
      step_times_ms_.begin() + (step_times_ms_.size() - static_cast<size_t>(timing_histogram_window_)));
  }

  ++timing_samples_since_report_;
  if (timing_samples_since_report_ < timing_histogram_report_every_) {
    return;
  }
  timing_samples_since_report_ = 0;

  if (step_times_ms_.empty()) {
    return;
  }

  std::vector<int> counts(timing_bins_ms_.size() - 1, 0);
  for (const double v : step_times_ms_) {
    for (size_t i = 0; i + 1 < timing_bins_ms_.size(); ++i) {
      const double left = timing_bins_ms_[i];
      const double right = timing_bins_ms_[i + 1];
      if ((v >= left && v < right) || (i + 2 == timing_bins_ms_.size() && v >= left)) {
        counts[i]++;
        break;
      }
    }
  }

  std::ostringstream hist;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (i > 0) {
      hist << " | ";
    }
    const double left = timing_bins_ms_[i];
    const double right = timing_bins_ms_[i + 1];
    if (right >= 1e8) {
      hist << "[" << std::fixed << std::setprecision(1) << left << ",inf):" << counts[i];
    } else {
      hist << "[" << std::fixed << std::setprecision(1) << left << "," << right << "):" << counts[i];
    }
  }

  double sum = 0.0;
  for (const double v : step_times_ms_) {
    sum += v;
  }
  const double mean_ms = sum / static_cast<double>(step_times_ms_.size());
  const double p50 = percentile(step_times_ms_, 50.0);
  const double p90 = percentile(step_times_ms_, 90.0);
  const double p99 = percentile(step_times_ms_, 99.0);
  const double max_ms = *std::max_element(step_times_ms_.begin(), step_times_ms_.end());
  const double fps = (mean_ms > 1e-9) ? (1000.0 / mean_ms) : 0.0;

  RCLCPP_INFO(
    this->get_logger(),
    "step_time_ms histogram (rolling=%zu): %s mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f approx_fps=%.1f",
    step_times_ms_.size(), hist.str().c_str(), mean_ms, p50, p90, p99, max_ms, fps);
}

void GridGroundNode::cb_points(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  const auto t0 = std::chrono::steady_clock::now();

  try {
    const bool verbose = this->get_parameter("verbose_debug_logs").as_bool();
    const bool publish_raw_debug = this->get_parameter("publish_raw_debug_cloud").as_bool();
    const int raw_debug_max_points = std::max(
      0, static_cast<int>(this->get_parameter("raw_debug_max_points").as_int()));

    std::vector<Eigen::Vector3f> pts = pc2_to_xyz(*msg);

    if (verbose) {
      RCLCPP_INFO(this->get_logger(), "parsed pts=%zu", pts.size());
      RCLCPP_INFO(
        this->get_logger(), "cb fired: width=%u height=%u step=%u bigendian=%s",
        msg->width, msg->height, msg->point_step, msg->is_bigendian ? "true" : "false");
    }

    if (pts.empty()) {
      return;
    }

    if (publish_raw_debug && raw_debug_max_points > 0) {
      pub_rawdebug_->publish(make_xyz_cloud(msg->header, pts, static_cast<size_t>(raw_debug_max_points)));
    }

    const size_t n_raw = pts.size();
    std::vector<Eigen::Vector3f> pts_finite;
    pts_finite.reserve(pts.size());
    for (const auto & p : pts) {
      if (std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z())) {
        pts_finite.push_back(p);
      }
    }

    std::vector<Eigen::Vector3f> pts_sane;
    pts_sane.reserve(pts_finite.size());
    for (const auto & p : pts_finite) {
      if (std::fabs(p.x()) < 1e3F && std::fabs(p.y()) < 1e3F && std::fabs(p.z()) < 1e3F) {
        pts_sane.push_back(p);
      }
    }

    std::string up_axis = this->get_parameter("up_axis").as_string();
    std::transform(up_axis.begin(), up_axis.end(), up_axis.begin(), ::tolower);
    if (up_axis != "x" && up_axis != "y" && up_axis != "z") {
      RCLCPP_WARN(this->get_logger(), "Invalid up_axis='%s', falling back to 'x'. Valid: x|y|z", up_axis.c_str());
      up_axis = "x";
    }

    const int up_i = axis_index(up_axis);
    int a0 = 0;
    int a1 = 1;
    if (up_i == 0) {
      a0 = 1;
      a1 = 2;
    } else if (up_i == 1) {
      a0 = 0;
      a1 = 2;
    } else {
      a0 = 0;
      a1 = 1;
    }

    const std::array<double, 3> mins = {
      this->get_parameter("x_min").as_double(),
      this->get_parameter("y_min").as_double(),
      this->get_parameter("z_min").as_double()};

    const std::array<double, 3> maxs = {
      this->get_parameter("x_max").as_double(),
      this->get_parameter("y_max").as_double(),
      this->get_parameter("z_max").as_double()};

    const double min_range = this->get_parameter("min_range").as_double();
    const double max_range = this->get_parameter("max_range").as_double();
    const double min_r2 = min_range * min_range;
    const double max_r2 = max_range * max_range;

    std::vector<Eigen::Vector3f> filtered;
    filtered.reserve(pts_sane.size());
    for (const auto & p : pts_sane) {
      const double r2 = static_cast<double>(p.x()) * p.x() +
        static_cast<double>(p.y()) * p.y() +
        static_cast<double>(p.z()) * p.z();
      if (r2 < min_r2 || r2 > max_r2) {
        continue;
      }
      if (p[a0] < mins[a0] || p[a0] > maxs[a0] || p[a1] < mins[a1] || p[a1] > maxs[a1]) {
        continue;
      }
      filtered.push_back(p);
    }

    if (verbose) {
      RCLCPP_INFO(
        this->get_logger(), "gates: raw=%zu finite=%zu sane=%zu after(range+roi)=%zu",
        n_raw, pts_finite.size(), pts_sane.size(), filtered.size());
    }

    if (filtered.empty()) {
      RCLCPP_WARN(this->get_logger(), "No points left after filtering (range/ROI likely wrong).");
      return;
    }

    const double cell_size = this->get_parameter("cell_size").as_double();
    const int min_points_per_cell = this->get_parameter("min_points_per_cell").as_int();
    const float outlier_dist = static_cast<float>(this->get_parameter("outlier_dist").as_double());
    const bool prefer_pos_z = this->get_parameter("prefer_normal_positive_z").as_bool();

    std::unordered_map<CellCoord, std::vector<size_t>, CellCoordHash> cell_to_indices;
    cell_to_indices.reserve(filtered.size());

    for (size_t i = 0; i < filtered.size(); ++i) {
      const int ix = static_cast<int>(std::floor((filtered[i][a0] - mins[a0]) / cell_size));
      const int iy = static_cast<int>(std::floor((filtered[i][a1] - mins[a1]) / cell_size));
      cell_to_indices[{ix, iy}].push_back(i);
    }

    std::vector<CellOutput> out;
    out.reserve(cell_to_indices.size());

    int fit_ok = 0;
    int fit_fail = 0;
    int nx_reject = 0;
    double total_inlier = 0.0;

    for (const auto & kv : cell_to_indices) {
      const CellCoord coord = kv.first;
      const std::vector<size_t> & idxs = kv.second;

      if (static_cast<int>(idxs.size()) < min_points_per_cell) {
        continue;
      }

      std::vector<Eigen::Vector3f> cell_pts;
      cell_pts.reserve(idxs.size());
      for (const auto idx : idxs) {
        cell_pts.push_back(filtered[idx]);
      }

      Eigen::Vector3f centroid;
      Eigen::Vector3f normal;
      float inlier_ratio = 0.0F;
      if (!fit_plane_pca_robust(cell_pts, outlier_dist, centroid, normal, inlier_ratio)) {
        ++fit_fail;
        continue;
      }

      ++fit_ok;
      total_inlier += inlier_ratio;

      if (prefer_pos_z) {
        if (normal[up_i] > 0.0F) {
          normal = -normal;
        }
      } else {
        if (normal[up_i] < 0.0F) {
          normal = -normal;
        }
      }

      const float p0_c = static_cast<float>(mins[a0] + (static_cast<double>(coord.x) + 0.5) * cell_size);
      const float p1_c = static_cast<float>(mins[a1] + (static_cast<double>(coord.y) + 0.5) * cell_size);

      const float n_up = normal[up_i];
      const float n0 = normal[a0];
      const float n1 = normal[a1];
      if (std::fabs(n_up) < 1e-6F) {
        ++nx_reject;
        continue;
      }

      const float up_c = centroid[up_i] -
        (n0 * (p0_c - centroid[a0]) + n1 * (p1_c - centroid[a1])) / n_up;

      Eigen::Vector3f center(0.0F, 0.0F, 0.0F);
      center[up_i] = up_c;
      center[a0] = p0_c;
      center[a1] = p1_c;

      CellOutput c;
      c.center = center;
      c.normal = normal;
      c.confidence = std::clamp(inlier_ratio, 0.0F, 1.0F);
      c.cx = coord.x;
      c.cy = coord.y;
      out.push_back(c);
    }

    if (verbose) {
      const double avg_inlier = fit_ok > 0 ? (total_inlier / static_cast<double>(fit_ok)) : 0.0;
      RCLCPP_INFO(
        this->get_logger(),
        "plane: considered=%zu fit_ok=%d fit_fail=%d nx_reject=%d avg_inlier=%.3f outputs=%zu",
        cell_to_indices.size(), fit_ok, fit_fail, nx_reject, avg_inlier, out.size());
    }

    if (out.empty()) {
      return;
    }

    const double d_max = this->get_parameter("d_max").as_double();
    const bool use_8_neighbors = this->get_parameter("use_8_neighbors").as_bool();
    const int keep_k = std::max(
      1, static_cast<int>(this->get_parameter("ground_components_keep").as_int()));

    std::unordered_map<CellCoord, size_t, CellCoordHash> coord_to_idx;
    coord_to_idx.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      coord_to_idx[{out[i].cx, out[i].cy}] = i;
    }

    std::vector<std::pair<int, int>> neigh = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    if (use_8_neighbors) {
      neigh.push_back({-1, -1});
      neigh.push_back({-1, 1});
      neigh.push_back({1, -1});
      neigh.push_back({1, 1});
    }

    UnionFind uf(out.size());
    const double d_max2 = d_max * d_max;

    for (size_t i = 0; i < out.size(); ++i) {
      const int cx = out[i].cx;
      const int cy = out[i].cy;
      for (const auto & dxy : neigh) {
        CellCoord key{cx + dxy.first, cy + dxy.second};
        auto it = coord_to_idx.find(key);
        if (it == coord_to_idx.end()) {
          continue;
        }
        const size_t j = it->second;
        const Eigen::Vector3f diff = out[i].center - out[j].center;
        const double dist2 = static_cast<double>(diff.dot(diff));
        if (dist2 < d_max2) {
          uf.unite(static_cast<int>(i), static_cast<int>(j));
        }
      }
    }

    std::unordered_map<int, int> root_counts;
    root_counts.reserve(out.size());
    std::vector<int> roots(out.size(), 0);

    for (size_t i = 0; i < out.size(); ++i) {
      roots[i] = uf.find(static_cast<int>(i));
      root_counts[roots[i]]++;
    }

    std::vector<std::pair<int, int>> root_count_pairs(root_counts.begin(), root_counts.end());
    std::sort(root_count_pairs.begin(), root_count_pairs.end(), [](const auto & a, const auto & b) {
      return a.second > b.second;
    });

    std::unordered_set<int> keep_roots;
    for (int i = 0; i < std::min<int>(keep_k, root_count_pairs.size()); ++i) {
      keep_roots.insert(root_count_pairs[static_cast<size_t>(i)].first);
    }

    std::vector<CellOutput> out_ground;
    std::vector<CellOutput> out_nonground;
    out_ground.reserve(out.size());
    out_nonground.reserve(out.size());

    for (size_t i = 0; i < out.size(); ++i) {
      if (keep_roots.count(roots[i]) > 0) {
        out_ground.push_back(out[i]);
      } else {
        out_nonground.push_back(out[i]);
      }
    }

    if (!out_ground.empty()) {
      pub_ground_->publish(make_result_cloud(msg->header, out_ground));
    }
    if (!out_nonground.empty()) {
      pub_nonground_->publish(make_result_cloud(msg->header, out_nonground));
    }

    if (verbose) {
      RCLCPP_INFO(
        this->get_logger(), "publish: ground N=%zu nonground N=%zu",
        out_ground.size(), out_nonground.size());
    }

    if (this->get_parameter("publish_normals_markers").as_bool()) {
      const float length = static_cast<float>(this->get_parameter("normal_length").as_double());
      if (!out_ground.empty()) {
        pub_ground_normals_->publish(
          make_normal_marker_array(msg->header, out_ground, length, "ground_normals", {0.0F, 1.0F, 0.0F, 1.0F}));
      }
      if (!out_nonground.empty()) {
        pub_nonground_normals_->publish(
          make_normal_marker_array(msg->header, out_nonground, length, "nonground_normals", {1.0F, 0.0F, 0.0F, 1.0F}));
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Exception in cb_points: %s", e.what());
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double dt_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();
  record_step_time(dt_ms);
}

}  // namespace tof_ground_seg_cpp
