#include "tof_ground_seg_cpp/grid_ground_node.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace tof_ground_seg_cpp
{
namespace
{

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

struct AxisInfo
{
  int up = 2;
  int plane0 = 0;
  int plane1 = 1;
  std::string name = "z";
};

struct PlaneFitResult
{
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  float inlier_ratio = 0.0F;
};

struct MeshVertex
{
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  float confidence = 0.0F;
};

struct VertexAccumulator
{
  double weighted_height_sum = 0.0;
  double weight_sum = 0.0;
  float confidence_sum = 0.0F;
  std::vector<float> predicted_heights;
  float height = 0.0F;
  bool valid = false;
};

class UnionFind
{
public:
  explicit UnionFind(std::size_t size)
  : parent_(size), size_(size, 1)
  {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  std::size_t find(std::size_t index)
  {
    while (parent_[index] != index) {
      parent_[index] = parent_[parent_[index]];
      index = parent_[index];
    }
    return index;
  }

  void unite(std::size_t a, std::size_t b)
  {
    auto ra = find(a);
    auto rb = find(b);
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
  std::vector<std::size_t> parent_;
  std::vector<std::size_t> size_;
};

struct IntPairHash
{
  std::size_t operator()(const std::pair<int, int> & value) const noexcept
  {
    const auto a = static_cast<std::uint32_t>(value.first);
    const auto b = static_cast<std::uint32_t>(value.second);
    return (static_cast<std::size_t>(a) << 32) ^ static_cast<std::size_t>(b);
  }
};

AxisInfo get_axis_info(const std::string & up_axis)
{
  if (up_axis == "x") {
    return AxisInfo{0, 1, 2, "x"};
  }
  if (up_axis == "y") {
    return AxisInfo{1, 0, 2, "y"};
  }
  return AxisInfo{2, 0, 1, "z"};
}

std::string normalize_axis_name(std::string value)
{
  value.erase(
    value.begin(),
    std::find_if(value.begin(), value.end(), [](unsigned char c) {return !std::isspace(c);}));
  value.erase(
    std::find_if(value.rbegin(), value.rend(), [](unsigned char c) {return !std::isspace(c);}).base(),
    value.end());
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  return value;
}

float read_float32(const std::uint8_t * data, bool little_endian)
{
  std::array<std::uint8_t, 4> bytes{};
  if (little_endian) {
    std::copy(data, data + 4, bytes.begin());
  } else {
    std::reverse_copy(data, data + 4, bytes.begin());
  }
  float value = 0.0F;
  std::memcpy(&value, bytes.data(), sizeof(float));
  return value;
}

double score_points(const std::vector<Eigen::Vector3f> & points)
{
  if (points.empty()) {
    return 0.0;
  }

  std::size_t reasonable = 0;
  std::size_t finite = 0;
  for (const auto & point : points) {
    if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
      continue;
    }
    ++finite;
    if (
      std::fabs(point.x()) < 100.0F && std::fabs(point.y()) < 100.0F &&
      std::fabs(point.z()) < 100.0F)
    {
      ++reasonable;
    }
  }

  if (finite == 0U) {
    return 0.0;
  }
  return static_cast<double>(reasonable) / static_cast<double>(finite);
}

std::vector<Eigen::Vector3f> decode_xyz_points(
  const PointCloud2 & msg,
  bool little_endian,
  std::size_t limit_points = std::numeric_limits<std::size_t>::max())
{
  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  for (const auto & field : msg.fields) {
    if (field.name == "x") {
      x_offset = static_cast<int>(field.offset);
    } else if (field.name == "y") {
      y_offset = static_cast<int>(field.offset);
    } else if (field.name == "z") {
      z_offset = static_cast<int>(field.offset);
    }
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    throw std::runtime_error("PointCloud2 message is missing x/y/z fields");
  }

  const std::size_t point_count = std::min<std::size_t>(
    static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height),
    msg.data.size() / std::max<std::size_t>(1U, msg.point_step));
  const std::size_t count = std::min(point_count, limit_points);

  std::vector<Eigen::Vector3f> points;
  points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto * base = &msg.data[i * msg.point_step];
    points.emplace_back(
      read_float32(base + x_offset, little_endian),
      read_float32(base + y_offset, little_endian),
      read_float32(base + z_offset, little_endian));
  }
  return points;
}

std::vector<Eigen::Vector3f> pointcloud2_to_xyz(const PointCloud2 & msg)
{
  if (msg.width == 0U || msg.height == 0U || msg.point_step == 0U || msg.data.empty()) {
    return {};
  }

  const std::size_t sample_count = std::min<std::size_t>(
    5000U,
    static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height));
  const auto sample_little = decode_xyz_points(msg, true, sample_count);
  const auto sample_big = decode_xyz_points(msg, false, sample_count);
  const bool use_little = score_points(sample_little) >= score_points(sample_big);

  std::vector<Eigen::Vector3f> points = decode_xyz_points(msg, use_little);
  points.erase(
    std::remove_if(
      points.begin(), points.end(),
      [](const Eigen::Vector3f & point) {
        return
          !std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z()) ||
          std::fabs(point.x()) >= 1.0e3F || std::fabs(point.y()) >= 1.0e3F ||
          std::fabs(point.z()) >= 1.0e3F;
      }),
    points.end());
  return points;
}

PointCloud2 create_cloud_xyz(const std_msgs::msg::Header & header, const std::vector<Eigen::Vector3f> & points)
{
  PointCloud2 cloud;
  cloud.header = header;
  cloud.height = 1U;
  cloud.width = static_cast<std::uint32_t>(points.size());
  cloud.is_bigendian = false;
  cloud.is_dense = false;
  cloud.point_step = 12U;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.fields.resize(3);
  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0U;
  cloud.fields[0].datatype = PointField::FLOAT32;
  cloud.fields[0].count = 1U;
  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4U;
  cloud.fields[1].datatype = PointField::FLOAT32;
  cloud.fields[1].count = 1U;
  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8U;
  cloud.fields[2].datatype = PointField::FLOAT32;
  cloud.fields[2].count = 1U;
  cloud.data.resize(cloud.row_step);

  for (std::size_t i = 0; i < points.size(); ++i) {
    auto * base = cloud.data.data() + i * cloud.point_step;
    const float x = points[i].x();
    const float y = points[i].y();
    const float z = points[i].z();
    std::memcpy(base + 0, &x, sizeof(float));
    std::memcpy(base + 4, &y, sizeof(float));
    std::memcpy(base + 8, &z, sizeof(float));
  }
  return cloud;
}

PointCloud2 create_plane_cloud(
  const std_msgs::msg::Header & header,
  const std::vector<GridCellPlane> & cells)
{
  PointCloud2 cloud;
  cloud.header = header;
  cloud.height = 1U;
  cloud.width = static_cast<std::uint32_t>(cells.size());
  cloud.is_bigendian = false;
  cloud.is_dense = false;
  cloud.point_step = 28U;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.fields.resize(7);

  const std::array<std::string, 7> names{{"x", "y", "z", "nx", "ny", "nz", "confidence"}};
  for (std::size_t i = 0; i < names.size(); ++i) {
    cloud.fields[i].name = names[i];
    cloud.fields[i].offset = static_cast<std::uint32_t>(i * 4U);
    cloud.fields[i].datatype = PointField::FLOAT32;
    cloud.fields[i].count = 1U;
  }
  cloud.data.resize(cloud.row_step);

  for (std::size_t i = 0; i < cells.size(); ++i) {
    auto * base = cloud.data.data() + i * cloud.point_step;
    const std::array<float, 7> values{{
      cells[i].center.x(), cells[i].center.y(), cells[i].center.z(),
      cells[i].normal.x(), cells[i].normal.y(), cells[i].normal.z(),
      cells[i].confidence
    }};
    for (std::size_t j = 0; j < values.size(); ++j) {
      std::memcpy(base + j * 4U, &values[j], sizeof(float));
    }
  }
  return cloud;
}

MarkerArray create_normal_markers(
  const std_msgs::msg::Header & header,
  const std::vector<GridCellPlane> & cells,
  double length,
  const std::string & ns,
  const std::array<float, 4> & rgba)
{
  MarkerArray array;
  array.markers.reserve(cells.size());

  for (std::size_t i = 0; i < cells.size(); ++i) {
    Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = static_cast<int>(i);
    marker.type = Marker::ARROW;
    marker.action = Marker::ADD;

    geometry_msgs::msg::Point start;
    start.x = cells[i].center.x();
    start.y = cells[i].center.y();
    start.z = cells[i].center.z();

    geometry_msgs::msg::Point end;
    end.x = cells[i].center.x() + length * cells[i].normal.x();
    end.y = cells[i].center.y() + length * cells[i].normal.y();
    end.z = cells[i].center.z() + length * cells[i].normal.z();

    marker.points = {start, end};
    marker.scale.x = 0.01;
    marker.scale.y = 0.02;
    marker.scale.z = 0.03;
    marker.color.r = rgba[0];
    marker.color.g = rgba[1];
    marker.color.b = rgba[2];
    marker.color.a = rgba[3];
    array.markers.push_back(marker);
  }

  return array;
}

std::optional<PlaneFitResult> fit_plane_pca(const std::vector<Eigen::Vector3f> & points)
{
  if (points.size() < 3U) {
    return std::nullopt;
  }

  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto & point : points) {
    centroid += point;
  }
  centroid /= static_cast<float>(points.size());

  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto & point : points) {
    const Eigen::Vector3f delta = point - centroid;
    covariance += delta * delta.transpose();
  }
  covariance /= static_cast<float>(std::max<std::size_t>(1U, points.size() - 1U));

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return std::nullopt;
  }

  Eigen::Vector3f normal = solver.eigenvectors().col(0);
  const float norm = normal.norm();
  if (norm < 1.0e-9F) {
    normal = Eigen::Vector3f(0.0F, 0.0F, 1.0F);
  } else {
    normal /= norm;
  }

  return PlaneFitResult{centroid, normal, 1.0F};
}

std::optional<PlaneFitResult> fit_plane_pca_robust(
  const std::vector<Eigen::Vector3f> & points,
  double outlier_dist)
{
  auto initial_fit = fit_plane_pca(points);
  if (!initial_fit.has_value()) {
    return std::nullopt;
  }

  auto compute_inliers = [&](const PlaneFitResult & fit) {
      std::vector<Eigen::Vector3f> inliers;
      inliers.reserve(points.size());
      for (const auto & point : points) {
        const float distance = std::fabs((point - fit.centroid).dot(fit.normal));
        if (distance <= static_cast<float>(outlier_dist)) {
          inliers.push_back(point);
        }
      }
      const float ratio = static_cast<float>(inliers.size()) / static_cast<float>(points.size());
      return std::make_pair(std::move(inliers), ratio);
    };

  auto [inliers, ratio] = compute_inliers(*initial_fit);
  initial_fit->inlier_ratio = ratio;

  if (inliers.size() >= 3U && ratio < 0.999F) {
    auto refit = fit_plane_pca(inliers);
    if (refit.has_value()) {
      auto [refit_inliers, refit_ratio] = compute_inliers(*refit);
      (void)refit_inliers;
      refit->inlier_ratio = refit_ratio;
      return refit;
    }
  }

  return initial_fit;
}

double percentile(std::vector<double> values, double fraction)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const auto left = static_cast<std::size_t>(std::floor(position));
  const auto right = static_cast<std::size_t>(std::ceil(position));
  if (left == right) {
    return values[left];
  }
  const double alpha = position - static_cast<double>(left);
  return values[left] * (1.0 - alpha) + values[right] * alpha;
}

std::string format_histogram(const std::vector<double> & samples, const std::vector<double> & bins)
{
  if (samples.empty() || bins.size() < 2U) {
    return {};
  }

  std::vector<int> counts(bins.size() - 1U, 0);
  for (const double value : samples) {
    for (std::size_t i = 0; i + 1U < bins.size(); ++i) {
      const bool last_bin = (i + 2U == bins.size());
      if ((value >= bins[i] && value < bins[i + 1]) || (last_bin && value >= bins[i])) {
        ++counts[i];
        break;
      }
    }
  }

  std::ostringstream stream;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (i > 0U) {
      stream << " | ";
    }
    stream << "[" << bins[i] << ",";
    if (bins[i + 1] > 1.0e8) {
      stream << "inf):";
    } else {
      stream << bins[i + 1] << "):";
    }
    stream << counts[i];
  }
  return stream.str();
}

std::optional<float> predict_up_at_planar(
  const GridCellPlane & cell,
  const AxisInfo & axis,
  float plane0,
  float plane1)
{
  const float n_up = cell.normal[axis.up];
  if (std::fabs(n_up) < 1.0e-6F) {
    return std::nullopt;
  }

  const float predicted = cell.center[axis.up] -
    (cell.normal[axis.plane0] * (plane0 - cell.center[axis.plane0]) +
    cell.normal[axis.plane1] * (plane1 - cell.center[axis.plane1])) / n_up;
  return predicted;
}

geometry_msgs::msg::Point to_geometry_point(const Eigen::Vector3f & point)
{
  geometry_msgs::msg::Point out;
  out.x = point.x();
  out.y = point.y();
  out.z = point.z();
  return out;
}

Marker create_delete_marker(const std_msgs::msg::Header & header, const std::string & ns, int id)
{
  Marker marker;
  marker.header = header;
  marker.ns = ns;
  marker.id = id;
  marker.action = Marker::DELETE;
  return marker;
}

void append_oriented_triangle(
  Marker & marker,
  const AxisInfo & axis,
  const Eigen::Vector3f & a,
  const Eigen::Vector3f & b,
  const Eigen::Vector3f & c)
{
  Eigen::Vector3f front_a = a;
  Eigen::Vector3f front_b = b;
  Eigen::Vector3f front_c = c;

  const Eigen::Vector3f ab = b - a;
  const Eigen::Vector3f ac = c - a;
  const Eigen::Vector3f normal = ab.cross(ac);
  if (normal[axis.up] < 0.0F) {
    front_b = c;
    front_c = b;
  }

  marker.points.push_back(to_geometry_point(front_a));
  marker.points.push_back(to_geometry_point(front_b));
  marker.points.push_back(to_geometry_point(front_c));

  marker.points.push_back(to_geometry_point(front_a));
  marker.points.push_back(to_geometry_point(front_c));
  marker.points.push_back(to_geometry_point(front_b));
}
std::optional<Marker> create_ground_mesh_marker(
  const std_msgs::msg::Header & header,
  const std::vector<GridCellPlane> & ground_cells,
  const AxisInfo & axis,
  const GridGroundNode::RuntimeConfig & config)
{
  if (ground_cells.empty()) {
    return std::nullopt;
  }

  std::unordered_map<std::pair<int, int>, VertexAccumulator, IntPairHash> vertices;
  vertices.reserve(ground_cells.size() * 4U);

  for (const auto & cell : ground_cells) {
    const std::array<std::pair<int, int>, 4> corners{{
      {cell.cell_x, cell.cell_y},
      {cell.cell_x + 1, cell.cell_y},
      {cell.cell_x, cell.cell_y + 1},
      {cell.cell_x + 1, cell.cell_y + 1}
    }};

    for (const auto & corner : corners) {
      const float plane0 = static_cast<float>(
        config.mins[axis.plane0] + static_cast<double>(corner.first) * config.cell_size);
      const float plane1 = static_cast<float>(
        config.mins[axis.plane1] + static_cast<double>(corner.second) * config.cell_size);
      const auto predicted = predict_up_at_planar(cell, axis, plane0, plane1);
      if (!predicted.has_value()) {
        continue;
      }

      auto & vertex = vertices[corner];
      const double weight = std::max(0.05, static_cast<double>(cell.confidence));
      vertex.weighted_height_sum += weight * static_cast<double>(*predicted);
      vertex.weight_sum += weight;
      vertex.confidence_sum += cell.confidence;
      vertex.predicted_heights.push_back(*predicted);
      vertex.valid = true;
    }
  }

  if (vertices.empty()) {
    return std::nullopt;
  }

  for (auto & [corner, vertex] : vertices) {
    (void)corner;
    if (!vertex.valid || vertex.weight_sum <= 0.0 || vertex.predicted_heights.empty()) {
      vertex.valid = false;
      continue;
    }

    std::sort(vertex.predicted_heights.begin(), vertex.predicted_heights.end());
    const std::size_t mid = vertex.predicted_heights.size() / 2U;
    float median_height = vertex.predicted_heights[mid];
    if (vertex.predicted_heights.size() % 2U == 0U) {
      median_height = 0.5F * (vertex.predicted_heights[mid - 1U] + vertex.predicted_heights[mid]);
    }

    const float mean_height = static_cast<float>(vertex.weighted_height_sum / vertex.weight_sum);
    vertex.height = static_cast<float>(0.75 * static_cast<double>(median_height) + 0.25 * static_cast<double>(mean_height));
    vertex.confidence_sum = std::clamp(vertex.confidence_sum / static_cast<float>(vertex.weight_sum), 0.0F, 1.0F);
  }

  const std::array<std::pair<int, int>, 4> smooth_neighbors{{
    std::pair<int, int>{-1, 0}, {1, 0}, {0, -1}, {0, 1}
  }};
  for (int iter = 0; iter < config.mesh_smoothing_iterations; ++iter) {
    std::unordered_map<std::pair<int, int>, float, IntPairHash> next_heights;
    next_heights.reserve(vertices.size());

    for (const auto & [corner, vertex] : vertices) {
      if (!vertex.valid) {
        continue;
      }

      double neighbor_sum = 0.0;
      double neighbor_weight = 0.0;
      for (const auto & offset : smooth_neighbors) {
        const auto it = vertices.find({corner.first + offset.first, corner.second + offset.second});
        if (it == vertices.end() || !it->second.valid) {
          continue;
        }
        const double delta = std::fabs(static_cast<double>(vertex.height) - static_cast<double>(it->second.height));
        if (delta > config.mesh_edge_height_threshold) {
          continue;
        }
        neighbor_sum += static_cast<double>(it->second.height);
        neighbor_weight += 1.0;
      }

      float next_height = vertex.height;
      if (neighbor_weight > 0.0) {
        const double neighbor_mean = neighbor_sum / neighbor_weight;
        next_height = static_cast<float>(
          config.mesh_alpha * static_cast<double>(vertex.height) +
          (1.0 - config.mesh_alpha) * neighbor_mean);
      }
      next_heights.emplace(corner, next_height);
    }

    for (const auto & [corner, next_height] : next_heights) {
      vertices[corner].height = next_height;
    }
  }

  std::unordered_map<std::pair<int, int>, float, IntPairHash> corrected_heights;
  corrected_heights.reserve(vertices.size());
  for (const auto & [corner, vertex] : vertices) {
    if (!vertex.valid) {
      continue;
    }

    std::vector<float> neighbor_heights;
    neighbor_heights.reserve(smooth_neighbors.size());
    for (const auto & offset : smooth_neighbors) {
      const auto it = vertices.find({corner.first + offset.first, corner.second + offset.second});
      if (it == vertices.end() || !it->second.valid) {
        continue;
      }
      neighbor_heights.push_back(it->second.height);
    }

    if (neighbor_heights.size() < 2U) {
      continue;
    }

    std::sort(neighbor_heights.begin(), neighbor_heights.end());
    const std::size_t mid = neighbor_heights.size() / 2U;
    float neighbor_median = neighbor_heights[mid];
    if (neighbor_heights.size() % 2U == 0U) {
      neighbor_median = 0.5F * (neighbor_heights[mid - 1U] + neighbor_heights[mid]);
    }

    if (std::fabs(vertex.height - neighbor_median) <= static_cast<float>(config.mesh_spike_height_threshold)) {
      continue;
    }

    corrected_heights.emplace(
      corner,
      static_cast<float>(0.35 * static_cast<double>(vertex.height) +
      0.65 * static_cast<double>(neighbor_median)));
  }

  for (const auto & [corner, corrected_height] : corrected_heights) {
    vertices[corner].height = corrected_height;
  }

  auto make_vertex = [&](const std::pair<int, int> & corner) -> std::optional<MeshVertex> {
      const auto it = vertices.find(corner);
      if (it == vertices.end() || !it->second.valid) {
        return std::nullopt;
      }

      MeshVertex vertex;
      vertex.position[axis.up] = it->second.height;
      vertex.position[axis.plane0] = static_cast<float>(
        config.mins[axis.plane0] + static_cast<double>(corner.first) * config.cell_size);
      vertex.position[axis.plane1] = static_cast<float>(
        config.mins[axis.plane1] + static_cast<double>(corner.second) * config.cell_size);
      vertex.confidence = it->second.confidence_sum;
      return vertex;
    };

  Marker marker;
  marker.header = header;
  marker.ns = "ground_mesh";
  marker.id = 0;
  marker.type = Marker::TRIANGLE_LIST;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.r = 0.60F;
  marker.color.g = 0.95F;
  marker.color.b = 0.60F;
  marker.color.a = static_cast<float>(std::clamp(config.mesh_marker_alpha, 0.05, 1.0));
  marker.points.reserve(ground_cells.size() * 6U);

  for (const auto & cell : ground_cells) {
    const auto v00 = make_vertex({cell.cell_x, cell.cell_y});
    const auto v10 = make_vertex({cell.cell_x + 1, cell.cell_y});
    const auto v01 = make_vertex({cell.cell_x, cell.cell_y + 1});
    const auto v11 = make_vertex({cell.cell_x + 1, cell.cell_y + 1});
    if (!v00.has_value() || !v10.has_value() || !v01.has_value() || !v11.has_value()) {
      continue;
    }

    const float edge_00_10 = std::fabs(v00->position[axis.up] - v10->position[axis.up]);
    const float edge_10_11 = std::fabs(v10->position[axis.up] - v11->position[axis.up]);
    const float edge_11_01 = std::fabs(v11->position[axis.up] - v01->position[axis.up]);
    const float edge_01_00 = std::fabs(v01->position[axis.up] - v00->position[axis.up]);
    const float max_edge_step = std::max(std::max(edge_00_10, edge_10_11), std::max(edge_11_01, edge_01_00));
    if (max_edge_step > static_cast<float>(config.mesh_max_triangle_height_step)) {
      continue;
    }

    const float diag_a = std::fabs(v00->position[axis.up] - v11->position[axis.up]);
    const float diag_b = std::fabs(v10->position[axis.up] - v01->position[axis.up]);

    if (diag_a <= diag_b) {
      append_oriented_triangle(marker, axis, v00->position, v10->position, v11->position);
      append_oriented_triangle(marker, axis, v00->position, v11->position, v01->position);
    } else {
      append_oriented_triangle(marker, axis, v00->position, v10->position, v01->position);
      append_oriented_triangle(marker, axis, v10->position, v11->position, v01->position);
    }
  }

  if (marker.points.empty()) {
    return std::nullopt;
  }

  return marker;
}

}  // namespace

GridGroundNode::GridGroundNode()
: Node("tof_grid_ground"),
  timing_bins_ms_({0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 33.3, 50.0, 100.0, 200.0, 1.0e9})
{
  declare_parameters();

  const auto points_topic = this->get_parameter("points_topic").as_string();
  const auto ground_topic = this->get_parameter("ground_topic").as_string();
  const auto nonground_topic = this->get_parameter("nonground_topic").as_string();
  const auto ground_normals_topic = this->get_parameter("ground_normals_topic").as_string();
  const auto nonground_normals_topic = this->get_parameter("nonground_normals_topic").as_string();
  const auto ground_mesh_topic = this->get_parameter("ground_mesh_topic").as_string();

  raw_debug_pub_ = this->create_publisher<PointCloud2>("/debug/raw_points_xyz", 10);
  ground_pub_ = this->create_publisher<PointCloud2>(ground_topic, 10);
  nonground_pub_ = this->create_publisher<PointCloud2>(nonground_topic, 10);
  ground_mesh_pub_ = this->create_publisher<Marker>(ground_mesh_topic, 10);
  ground_normals_pub_ = this->create_publisher<MarkerArray>(ground_normals_topic, 10);
  nonground_normals_pub_ = this->create_publisher<MarkerArray>(nonground_normals_topic, 10);
  points_sub_ = this->create_subscription<PointCloud2>(
    points_topic, rclcpp::SensorDataQoS(),
    std::bind(&GridGroundNode::handle_points, this, std::placeholders::_1));

  timing_histogram_enabled_ = this->get_parameter("timing_histogram_enabled").as_bool();
  timing_histogram_window_ = std::max(
    1, static_cast<int>(this->get_parameter("timing_histogram_window").as_int()));
  timing_histogram_report_every_ = std::max(
    1, static_cast<int>(this->get_parameter("timing_histogram_report_every").as_int()));
  timing_log_each_step_ = this->get_parameter("timing_log_each_step").as_bool();
  timing_save_txt_ = this->get_parameter("timing_save_txt").as_bool();
  timing_txt_path_ = this->get_parameter("timing_txt_path").as_string();

  if (timing_save_txt_ && !timing_txt_path_.empty()) {
    const auto parent = std::filesystem::path(timing_txt_path_).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    timing_txt_file_ = std::make_unique<std::ofstream>(timing_txt_path_, std::ios::app);
  }

  RCLCPP_INFO(this->get_logger(), "Listening: %s", points_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing ground grid cells: %s", ground_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing non-ground grid cells: %s", nonground_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing ground mesh marker: %s", ground_mesh_topic.c_str());
}

GridGroundNode::~GridGroundNode()
{
  if (timing_txt_file_ && timing_txt_file_->is_open()) {
    timing_txt_file_->close();
  }
}

void GridGroundNode::declare_parameters()
{
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
  this->declare_parameter<bool>("publish_ground_mesh", true);
  this->declare_parameter<std::string>("ground_mesh_topic", "/ground_mesh");
  this->declare_parameter<int>("mesh_smoothing_iterations", 2);
  this->declare_parameter<double>("mesh_edge_height_threshold", 0.05);
  this->declare_parameter<double>("mesh_alpha", 0.85);
  this->declare_parameter<double>("mesh_marker_alpha", 0.85);
  this->declare_parameter<double>("mesh_spike_height_threshold", 0.08);
  this->declare_parameter<double>("mesh_max_triangle_height_step", 0.10);
  this->declare_parameter<bool>("timing_histogram_enabled", true);
  this->declare_parameter<int>("timing_histogram_window", 200);
  this->declare_parameter<int>("timing_histogram_report_every", 30);
  this->declare_parameter<bool>("timing_log_each_step", false);
  this->declare_parameter<bool>("timing_save_txt", true);
  this->declare_parameter<std::string>(
    "timing_txt_path", "/home/keqi/BFHRobotic/Plot/Real_time/tof_cpp_step_time_ms.txt");
  this->declare_parameter<bool>("publish_raw_debug_cloud", false);
  this->declare_parameter<int>("raw_debug_max_points", 20000);
  this->declare_parameter<bool>("verbose_debug_logs", false);
}

GridGroundNode::RuntimeConfig GridGroundNode::get_runtime_config() const
{
  RuntimeConfig config;
  config.up_axis = normalize_axis_name(this->get_parameter("up_axis").as_string());
  config.cell_size = this->get_parameter("cell_size").as_double();
  config.min_points_per_cell = static_cast<int>(this->get_parameter("min_points_per_cell").as_int());
  config.outlier_dist = this->get_parameter("outlier_dist").as_double();
  config.min_range = this->get_parameter("min_range").as_double();
  config.max_range = this->get_parameter("max_range").as_double();
  config.prefer_normal_positive_z = this->get_parameter("prefer_normal_positive_z").as_bool();
  config.d_max = this->get_parameter("d_max").as_double();
  config.use_8_neighbors = this->get_parameter("use_8_neighbors").as_bool();
  config.ground_components_keep = static_cast<int>(this->get_parameter("ground_components_keep").as_int());
  config.publish_normals_markers = this->get_parameter("publish_normals_markers").as_bool();
  config.normal_length = this->get_parameter("normal_length").as_double();
  config.publish_ground_mesh = this->get_parameter("publish_ground_mesh").as_bool();
  config.ground_mesh_topic = this->get_parameter("ground_mesh_topic").as_string();
  config.mesh_smoothing_iterations = static_cast<int>(this->get_parameter("mesh_smoothing_iterations").as_int());
  config.mesh_edge_height_threshold = this->get_parameter("mesh_edge_height_threshold").as_double();
  config.mesh_alpha = this->get_parameter("mesh_alpha").as_double();
  config.mesh_marker_alpha = this->get_parameter("mesh_marker_alpha").as_double();
  config.mesh_spike_height_threshold = this->get_parameter("mesh_spike_height_threshold").as_double();
  config.mesh_max_triangle_height_step = this->get_parameter("mesh_max_triangle_height_step").as_double();
  config.publish_raw_debug_cloud = this->get_parameter("publish_raw_debug_cloud").as_bool();
  config.raw_debug_max_points = static_cast<int>(this->get_parameter("raw_debug_max_points").as_int());
  config.verbose_debug_logs = this->get_parameter("verbose_debug_logs").as_bool();
  config.mins = {
    this->get_parameter("x_min").as_double(),
    this->get_parameter("y_min").as_double(),
    this->get_parameter("z_min").as_double()
  };
  config.maxs = {
    this->get_parameter("x_max").as_double(),
    this->get_parameter("y_max").as_double(),
    this->get_parameter("z_max").as_double()
  };
  return config;
}

void GridGroundNode::record_step_time(double dt_ms)
{
  if (timing_log_each_step_) {
    RCLCPP_INFO(this->get_logger(), "step_time_ms_sample=%.3f", dt_ms);
  }

  if (timing_txt_file_ && timing_txt_file_->is_open()) {
    (*timing_txt_file_) << dt_ms << '\n';
    timing_txt_file_->flush();
  }

  if (!timing_histogram_enabled_) {
    return;
  }

  step_times_ms_.push_back(dt_ms);
  if (static_cast<int>(step_times_ms_.size()) > timing_histogram_window_) {
    step_times_ms_.erase(
      step_times_ms_.begin(),
      step_times_ms_.begin() + (step_times_ms_.size() - static_cast<std::size_t>(timing_histogram_window_)));
  }

  ++timing_samples_since_report_;
  if (timing_samples_since_report_ < timing_histogram_report_every_) {
    return;
  }
  timing_samples_since_report_ = 0;

  const double mean = std::accumulate(step_times_ms_.begin(), step_times_ms_.end(), 0.0) /
    static_cast<double>(step_times_ms_.size());
  const double p50 = percentile(step_times_ms_, 0.50);
  const double p90 = percentile(step_times_ms_, 0.90);
  const double p99 = percentile(step_times_ms_, 0.99);
  const double maximum = *std::max_element(step_times_ms_.begin(), step_times_ms_.end());
  const double fps = (mean > 1.0e-9) ? (1000.0 / mean) : 0.0;

  RCLCPP_INFO(
    this->get_logger(),
    "step_time_ms histogram (rolling=%zu): %s mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f approx_fps=%.1f",
    step_times_ms_.size(),
    format_histogram(step_times_ms_, timing_bins_ms_).c_str(),
    mean,
    p50,
    p90,
    p99,
    maximum,
    fps);
}

void GridGroundNode::handle_points(const PointCloud2::SharedPtr msg)
{
  const auto start = std::chrono::steady_clock::now();
  try {
    const auto config = get_runtime_config();
    const std::string requested_axis = config.up_axis;
    AxisInfo axis = get_axis_info(config.up_axis);
    if (config.up_axis != "x" && config.up_axis != "y" && config.up_axis != "z") {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Invalid up_axis='%s', falling back to 'x'. Valid values: x|y|z",
        requested_axis.c_str());
      axis = get_axis_info("x");
    }

    std::vector<Eigen::Vector3f> points = pointcloud2_to_xyz(*msg);
    if (points.empty()) {
      return;
    }

    if (config.publish_raw_debug_cloud && config.raw_debug_max_points > 0) {
      const std::size_t count = std::min<std::size_t>(points.size(), static_cast<std::size_t>(config.raw_debug_max_points));
      std::vector<Eigen::Vector3f> sample(points.begin(), points.begin() + count);
      raw_debug_pub_->publish(create_cloud_xyz(msg->header, sample));
    }

    if (config.verbose_debug_logs) {
      Eigen::Vector3f min_point = points.front();
      Eigen::Vector3f max_point = points.front();
      for (const auto & point : points) {
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
      }
      RCLCPP_INFO(
        this->get_logger(),
        "parsed pts=%zu min(x,y,z)=[%.3f, %.3f, %.3f] max(x,y,z)=[%.3f, %.3f, %.3f] up_axis=%s",
        points.size(),
        min_point.x(), min_point.y(), min_point.z(),
        max_point.x(), max_point.y(), max_point.z(),
        axis.name.c_str());
    }

    std::vector<Eigen::Vector3f> filtered;
    filtered.reserve(points.size());
    const double min_range_sq = config.min_range * config.min_range;
    const double max_range_sq = config.max_range * config.max_range;
    for (const auto & point : points) {
      const double range_sq =
        static_cast<double>(point.x()) * point.x() +
        static_cast<double>(point.y()) * point.y() +
        static_cast<double>(point.z()) * point.z();
      if (range_sq < min_range_sq || range_sq > max_range_sq) {
        continue;
      }
      if (
        point[axis.plane0] < config.mins[axis.plane0] || point[axis.plane0] > config.maxs[axis.plane0] ||
        point[axis.plane1] < config.mins[axis.plane1] || point[axis.plane1] > config.maxs[axis.plane1])
      {
        continue;
      }
      filtered.push_back(point);
    }

    if (filtered.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "No points left after filtering. Check range or ROI parameters.");
      return;
    }

    if (config.verbose_debug_logs) {
      RCLCPP_INFO(
        this->get_logger(),
        "after filter pts=%zu plane_axes=(%d,%d) roi=[%.2f,%.2f]x[%.2f,%.2f] range=[%.2f,%.2f]",
        filtered.size(),
        axis.plane0,
        axis.plane1,
        config.mins[axis.plane0],
        config.maxs[axis.plane0],
        config.mins[axis.plane1],
        config.maxs[axis.plane1],
        config.min_range,
        config.max_range);
    }

    std::unordered_map<std::pair<int, int>, std::vector<Eigen::Vector3f>, IntPairHash> cell_points;
    cell_points.reserve(filtered.size());
    for (const auto & point : filtered) {
      const int ix = static_cast<int>(std::floor((point[axis.plane0] - config.mins[axis.plane0]) / config.cell_size));
      const int iy = static_cast<int>(std::floor((point[axis.plane1] - config.mins[axis.plane1]) / config.cell_size));
      cell_points[{ix, iy}].push_back(point);
    }

    std::vector<GridCellPlane> cells;
    cells.reserve(cell_points.size());
    int fit_ok = 0;
    int fit_fail = 0;
    int near_vertical_reject = 0;
    for (const auto & entry : cell_points) {
      if (static_cast<int>(entry.second.size()) < config.min_points_per_cell) {
        continue;
      }

      auto fit = fit_plane_pca_robust(entry.second, config.outlier_dist);
      if (!fit.has_value()) {
        ++fit_fail;
        continue;
      }
      ++fit_ok;

      Eigen::Vector3f normal = fit->normal;
      if (config.prefer_normal_positive_z) {
        if (normal[axis.up] > 0.0F) {
          normal = -normal;
        }
      } else if (normal[axis.up] < 0.0F) {
        normal = -normal;
      }

      const float n_up = normal[axis.up];
      if (std::fabs(n_up) < 1.0e-6F) {
        ++near_vertical_reject;
        continue;
      }

      const float p0_center = static_cast<float>(
        config.mins[axis.plane0] + (static_cast<double>(entry.first.first) + 0.5) * config.cell_size);
      const float p1_center = static_cast<float>(
        config.mins[axis.plane1] + (static_cast<double>(entry.first.second) + 0.5) * config.cell_size);
      const float up_center = fit->centroid[axis.up] -
        (normal[axis.plane0] * (p0_center - fit->centroid[axis.plane0]) +
        normal[axis.plane1] * (p1_center - fit->centroid[axis.plane1])) / n_up;

      GridCellPlane cell;
      cell.center[axis.up] = up_center;
      cell.center[axis.plane0] = p0_center;
      cell.center[axis.plane1] = p1_center;
      cell.normal = normal;
      cell.confidence = std::clamp(fit->inlier_ratio, 0.0F, 1.0F);
      cell.cell_x = entry.first.first;
      cell.cell_y = entry.first.second;
      cells.push_back(cell);
    }

    if (cells.empty()) {
      if (config.publish_ground_mesh) {
        ground_mesh_pub_->publish(create_delete_marker(msg->header, "ground_mesh", 0));
      }
      if (config.verbose_debug_logs) {
        RCLCPP_WARN(
          this->get_logger(),
          "No grid cells survived. occupied_cells=%zu fit_ok=%d fit_fail=%d near_vertical_reject=%d",
          cell_points.size(),
          fit_ok,
          fit_fail,
          near_vertical_reject);
      }
      return;
    }

    std::unordered_map<std::pair<int, int>, std::size_t, IntPairHash> index_by_cell;
    index_by_cell.reserve(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
      index_by_cell[{cells[i].cell_x, cells[i].cell_y}] = i;
    }

    std::vector<std::pair<int, int>> neighbors{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    if (config.use_8_neighbors) {
      neighbors.insert(neighbors.end(), {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}});
    }

    UnionFind uf(cells.size());
    const double d_max_sq = config.d_max * config.d_max;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      for (const auto & offset : neighbors) {
        const auto it = index_by_cell.find({cells[i].cell_x + offset.first, cells[i].cell_y + offset.second});
        if (it == index_by_cell.end()) {
          continue;
        }
        const Eigen::Vector3f delta = cells[i].center - cells[it->second].center;
        if (delta.squaredNorm() < static_cast<float>(d_max_sq)) {
          uf.unite(i, it->second);
        }
      }
    }

    std::unordered_map<std::size_t, int> component_sizes;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      ++component_sizes[uf.find(i)];
    }
    std::vector<std::pair<std::size_t, int>> ordered_components(component_sizes.begin(), component_sizes.end());
    std::sort(
      ordered_components.begin(), ordered_components.end(),
      [](const auto & lhs, const auto & rhs) {return lhs.second > rhs.second;});

    std::unordered_map<std::size_t, bool> keep_component;
    const int keep_count = std::max(1, config.ground_components_keep);
    for (int i = 0; i < keep_count && i < static_cast<int>(ordered_components.size()); ++i) {
      keep_component[ordered_components[i].first] = true;
    }

    std::vector<GridCellPlane> ground_cells;
    std::vector<GridCellPlane> nonground_cells;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      if (keep_component[uf.find(i)]) {
        ground_cells.push_back(cells[i]);
      } else {
        nonground_cells.push_back(cells[i]);
      }
    }

    if (!ground_cells.empty()) {
      ground_pub_->publish(create_plane_cloud(msg->header, ground_cells));
    }
    if (!nonground_cells.empty()) {
      nonground_pub_->publish(create_plane_cloud(msg->header, nonground_cells));
    }

    if (config.publish_ground_mesh) {
      const auto mesh_marker = create_ground_mesh_marker(msg->header, ground_cells, axis, config);
      if (mesh_marker.has_value()) {
        ground_mesh_pub_->publish(*mesh_marker);
      } else {
        ground_mesh_pub_->publish(create_delete_marker(msg->header, "ground_mesh", 0));
      }
    }

    if (config.publish_normals_markers) {
      if (!ground_cells.empty()) {
        ground_normals_pub_->publish(
          create_normal_markers(msg->header, ground_cells, config.normal_length, "ground_normals", {0.0F, 1.0F, 0.0F, 1.0F}));
      }
      if (!nonground_cells.empty()) {
        nonground_normals_pub_->publish(
          create_normal_markers(msg->header, nonground_cells, config.normal_length, "nonground_normals", {1.0F, 0.0F, 0.0F, 1.0F}));
      }
    }

    if (config.verbose_debug_logs) {
      RCLCPP_INFO(
        this->get_logger(),
        "processed input=%zu filtered=%zu occupied_cells=%zu fit_ok=%d fit_fail=%d reject=%d ground=%zu nonground=%zu",
        points.size(),
        filtered.size(),
        cell_points.size(),
        fit_ok,
        fit_fail,
        near_vertical_reject,
        ground_cells.size(),
        nonground_cells.size());
    }
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(this->get_logger(), "Point callback failed: %s", ex.what());
  }

  const auto elapsed = std::chrono::steady_clock::now() - start;
  record_step_time(std::chrono::duration<double, std::milli>(elapsed).count());
}

}  // namespace tof_ground_seg_cpp
