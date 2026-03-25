# Ground Mesh Marker Generation for RViz2

## Purpose

This note explains how the current system converts the input point cloud into a ground surface mesh marker for RViz2.

The goal of the pipeline is:

- detect local ground from the point cloud
- convert local ground estimates into a continuous surface
- publish the surface as triangles for visualization in RViz2
- reduce visual cracks, spikes, and unrealistic jumps

## 1. Input Point Cloud and Filtering

The node starts from an input `sensor_msgs/msg/PointCloud2`.

Before any ground estimation, the point cloud is filtered by:

- valid finite XYZ values
- minimum and maximum range
- ROI limits in the selected working axes
- chosen `up_axis`

This removes points that are too far, too close, outside the working region, or invalid.

## 2. Grid-Based Spatial Partition

The filtered point cloud is projected into a 2D grid.

The grid is built on the two axes orthogonal to the selected `up_axis`:

- if `up_axis = z`, the grid lies on `(x, y)`
- if `up_axis = x`, the grid lies on `(y, z)`
- if `up_axis = y`, the grid lies on `(x, z)`

Each point is assigned to one square cell using:

- `cell_x = floor((plane0 - min_plane0) / cell_size)`
- `cell_y = floor((plane1 - min_plane1) / cell_size)`

So the scene becomes a set of small grid cells, each containing a local subset of points.

## 3. Local Plane Estimation Per Grid Cell

For each occupied grid cell:

- if there are not enough points, the cell is rejected
- otherwise a robust PCA plane fit is computed
- outliers are reduced using an inlier-distance threshold

For each accepted cell, the algorithm stores:

- a cell center point in 3D
- a cell normal vector
- a confidence score
- the integer grid index `(cell_x, cell_y)`

This gives a local ground-plane hypothesis for every usable square cell.

## 4. Ground Cell Selection

The fitted cells are not all treated as ground automatically.

Neighboring cells are connected if their 3D centers are close enough. A connected-components step then keeps the largest component(s) as ground.

This suppresses isolated false detections and keeps the main continuous ground region.

After this step, the node has:

- `ground_cells`
- `nonground_cells`

The mesh marker is built only from `ground_cells`.

## 5. From Square Cells to Shared Mesh Vertices

The key idea is that the final surface is not built from independent cell patches.
Instead, neighboring cells share corner vertices.

For each ground cell, the four grid corners are:

- `(cell_x, cell_y)`
- `(cell_x + 1, cell_y)`
- `(cell_x, cell_y + 1)`
- `(cell_x + 1, cell_y + 1)`

Each corner becomes a candidate mesh vertex.

Because adjacent cells touch the same corners, they naturally share vertices. This is what makes the surface continuous.

## 6. Vertex Height Estimation

Each cell plane predicts a height at each of its four corners.

So for one shared corner, several neighboring cells may contribute predicted heights.

The system stores for each corner:

- all predicted heights from neighboring cells
- weighted sum of predicted heights
- total weight
- accumulated confidence

The initial vertex height is not taken from only one cell.
It is estimated from all available corner predictions.

### Robust Corner Height

To reduce failures from one bad local plane, the current system uses a robust corner estimate:

- sort all predicted heights for that corner
- compute the median predicted height
- compute the weighted mean predicted height
- combine them with a median-heavy blend

Current idea:

- `vertex_height = 0.75 * median + 0.25 * mean`

Why this helps:

- the median resists a single bad prediction
- the mean still preserves some smooth averaging behavior
- this prevents the surface from sagging too much below nearby flat ground

## 7. Mesh Smoothing on Shared Vertices

After the initial corner heights are estimated, the mesh is smoothed over the grid vertices.

This smoothing works only between neighboring vertices and only if the height difference is not too large.

For each valid vertex:

- gather the 4-neighbor vertices
- ignore invalid neighbors
- ignore neighbors whose height difference is larger than `mesh_edge_height_threshold`
- average the allowed neighbors
- blend current height with neighbor mean using `mesh_alpha`

This produces smoother ramps and flatter surfaces while avoiding smoothing across clear jumps.

## 8. Spike Suppression

Even after smoothing, an isolated corner can still be too high or too low.

To correct this, the system performs a spike check:

- collect neighboring vertex heights
- compute the neighbor median
- compare the current vertex against that median
- if the difference is larger than `mesh_spike_height_threshold`, pull the vertex back toward the local median

This is especially useful when a single local cell fit is corrupted by noise or an obstacle edge.

## 9. Building Triangles From the Grid

Each valid square cell is converted into two triangles.

A square has four corners:

- `v00`
- `v10`
- `v01`
- `v11`

There are two possible diagonals:

- `v00 -> v11`
- `v10 -> v01`

The system chooses the diagonal with the smaller vertical mismatch:

- compare `|v00.up - v11.up|`
- compare `|v10.up - v01.up|`
- choose the smaller one

This reduces folding artifacts and usually gives a more natural surface.

## 10. Edge Handling and Triangle Rejection

Not every square should become triangles.

Before generating triangles, the system checks the edge height differences:

- `|v00 - v10|`
- `|v10 - v11|`
- `|v11 - v01|`
- `|v01 - v00|`

If the largest edge step is greater than `mesh_max_triangle_height_step`, the square is skipped.

Why this is important:

- it prevents vertical walls from appearing in the mesh
- it avoids unrealistic triangles across strong discontinuities
- it reduces artifacts at obstacle boundaries

So the triangle stage is not just visualization. It is also a geometric quality filter.

## 11. Triangle Orientation

Each triangle is added to the RViz2 marker with a consistent winding order.

The triangle normal is computed from the cross product of two triangle edges.
If the normal points in the wrong direction relative to `up_axis`, the vertex order is flipped.

Why this matters:

- RViz2 shades triangle front and back faces differently
- inconsistent winding can make one side look green and the other side look black
- consistent orientation makes the visible surface shading more stable

## 12. Marker Publishing

The final surface is published as:

- type: `visualization_msgs/msg/Marker`
- marker type: `TRIANGLE_LIST`
- topic: `/ground_mesh` by default

Each triangle contributes 3 points to the marker.
A square cell contributes 2 triangles, so 6 points.

The marker also contains:

- fixed color
- alpha value
- namespace and ID
- identity pose

This is what RViz2 renders as the ground surface mesh.

## 13. Summary of the Full Pipeline

The full pipeline is:

1. read and decode point cloud
2. filter by range, ROI, and validity
3. assign points into 2D grid cells
4. fit one local plane per usable cell
5. keep the connected ground cells
6. convert cell planes into shared corner-height predictions
7. estimate robust corner heights
8. smooth neighboring vertices
9. suppress isolated spikes
10. reject cells with excessive edge height jump
11. split each square into two triangles
12. orient triangles consistently
13. publish the result as an RViz2 triangle mesh marker

## 14. Why This Method Was Chosen

Compared with publishing one plane per cell, this method gives:

- shared vertices between neighboring cells
- a more continuous surface
- fewer visual cracks
- smoother ramps and flat ground regions
- better suppression of sudden jumps

It is still grid-based, so it remains computationally simple and fits the original ground-segmentation design.

## 15. Short PPT Version

A short version suitable for slides is:

- The point cloud is filtered and divided into small 2D grid cells.
- A robust local plane is fitted inside each valid cell.
- Ground cells are selected by spatial continuity.
- Neighboring cell planes predict heights at shared grid corners.
- Corner heights are estimated robustly using a median-guided fusion.
- The shared corner heights are smoothed and corrected for spikes.
- Each square cell is split into two triangles using the better diagonal.
- Cells with excessive edge height change are rejected.
- Triangles are oriented consistently and published as a `TRIANGLE_LIST` marker in RViz2.
