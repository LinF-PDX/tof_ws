# Continuous Ground Surface Options for `tof_ground_seg_cpp`

## Current Behavior

The current node builds a 2D grid in the ground plane, fits one local plane per occupied cell, and then groups neighboring cells if their fitted centers are close enough.

That means:

- continuity is decided after fitting, not during surface construction
- each cell owns its own center height and normal
- neighboring cells do not share geometric state
- visible seams are expected when adjacent plane fits differ

This is good for simple segmentation, but it does not produce a continuous ground surface.

## What "Continuous" Should Mean

For this problem, "continuous" should usually mean:

- nearby ground regions share the same surface geometry
- smooth slopes stay smooth across cell borders
- small fitting noise does not create cracks between cells
- true discontinuities such as curbs or step edges are still allowed

So the target is not "force everything to be smooth". The target is:

- smooth where the scene supports it
- broken only where there is evidence of a real edge

## Method Comparison

### 1. Shared-Vertex Grid Mesh

Keep the existing 2D grid, but stop treating each square as an independent patch. Instead:

- define one height value at each grid corner
- each square becomes two triangles
- neighboring squares share corner vertices

Advantages:

- easiest transition from the current pipeline
- guaranteed positional continuity across cells
- predictable runtime and memory
- naturally fits the existing integer grid indexing

Disadvantages:

- triangle orientation matters near sharp edges
- still assumes the ground is mostly a height field over the chosen plane axes
- needs a strategy for empty corners and sparse regions

This is the strongest practical recommendation for the current codebase.

### 2. Overlapping Plane Fits With Blending

Keep cell planes, but evaluate the ground surface by blending several nearby fits.

Examples:

- fit each cell using the cell plus neighbors
- evaluate height from the 4 surrounding cells
- blend with distance or confidence weights

Advantages:

- small change in concepts
- removes hard jumps at cell borders
- robust to moderate noise

Disadvantages:

- not a true mesh
- continuity depends on the blending rule
- can smear sharp terrain transitions

This is the best minimal-change conceptual extension.

### 3. Direct Triangulation of Ground Points

After finding likely ground points, triangulate them in 2D and keep only triangles that pass geometric checks.

Examples:

- Delaunay triangulation in the plane coordinates
- reject triangles with long edges, high slope mismatch, or large residuals

Advantages:

- flexible geometry
- naturally triangle-based
- does not require a regular grid

Disadvantages:

- more sensitive to sparse ToF sampling
- more work to keep stable over holes and outliers
- harder to make deterministic frame to frame

This is better when the data density is good and the terrain is not well modeled by a regular grid.

### 4. Global Smooth Surface Fit

Fit one continuous surface across the ROI.

Examples:

- spline surface
- regularized height map
- moving least squares

Advantages:

- very smooth output
- good denoising

Disadvantages:

- easiest way to oversmooth curbs and local breaks
- more tuning effort
- less transparent behavior than a local grid mesh

This is usually the cleanest mathematically, but not the safest first step for a robotic ground detector.

### 5. Optimization Over Existing Cells

Keep one estimate per cell, but solve for neighboring consistency jointly.

Examples:

- data term keeps each cell near its local fit
- smoothness term penalizes height and normal disagreement
- robust penalty preserves real edges

Advantages:

- closest to the current implementation
- can improve continuity substantially

Disadvantages:

- still cell-centered rather than vertex-shared
- continuity is indirect
- requires solver design and tuning

This is useful if preserving the current pipeline is more important than having a clean geometric representation.

## Recommended Algorithm

## Name

Shared-Vertex Height-Field Triangle Mesh with Edge-Aware Smoothing

## Why This Fits the Current Node

The current system already has:

- a regular 2D grid
- per-cell point sets
- per-cell local plane fits
- neighbor relations on the grid

So the cleanest path is to replace "one independent surface per cell" with "one shared set of corner vertices for the whole ground region".

## Core Idea

Represent the ground as a height field over the two non-up axes.

Steps:

1. Build the same regular grid as today.
2. For each occupied cell, estimate a reliable local plane and confidence.
3. Introduce one vertex height at every grid corner.
4. Infer each vertex height from nearby cell planes and/or nearby points.
5. Triangulate each valid square into two triangles using those shared corner heights.
6. Apply edge-aware smoothing so smooth terrain becomes continuous while true breaks remain allowed.

Because adjacent squares share the same corner vertices, the surface becomes position-continuous by construction.

## Detailed Pipeline

### Step 1. Keep the Existing Pre-Filter

Retain the current logic for:

- range filtering
- ROI filtering
- axis selection
- point validity checks

This keeps the input stable and preserves the current deployment assumptions.

### Step 2. Fit Local Cell Planes

For each occupied cell:

- gather points in that cell
- reject cells with too few points
- fit a robust local plane
- compute confidence from inlier ratio, point count, and local spread

Do not treat this plane as the final geometry. Treat it as a local measurement.

### Step 3. Create a Grid of Shared Vertices

If the cell grid is `Nx` by `Ny`, create a corner-vertex grid of `(Nx + 1)` by `(Ny + 1)`.

Each vertex stores:

- planar coordinates from the grid geometry
- unknown height along the up axis
- confidence / support count

Now every square references four shared vertices:

- `v00`
- `v10`
- `v01`
- `v11`

### Step 4. Estimate Vertex Heights

For each vertex, gather evidence from the cells touching that corner.

For a regular interior vertex, this is up to four adjacent cells.

Each adjacent cell plane predicts a height at the vertex planar location. Combine those predictions with weights based on:

- plane confidence
- point count
- distance from cell centroid
- plane tilt sanity

A good first estimator is a weighted median or weighted average of predicted heights.

Weighted median is more robust.
Weighted average is smoother.

If a vertex has weak support:

- fall back to nearby raw points
- or interpolate from neighboring strong vertices
- or mark it invalid and avoid building triangles that depend on it

### Step 5. Edge-Aware Smoothing of Vertex Heights

Once the initial vertex heights are estimated, run a local smoothing stage.

The smoothing should:

- reduce small frame noise
- improve continuity on ramps
- preserve true terrain edges

A practical formulation is:

- data term: keep each vertex near its estimated height
- smoothness term: neighboring vertices should have similar heights and slopes
- robust edge term: weaken smoothing if adjacent supporting cell planes disagree strongly

Signals that should reduce smoothing:

- large predicted height jump
- large normal-angle difference
- low confidence across the boundary
- missing data on one side

This is important. Without edge awareness, the method will smear curbs.

### Step 6. Triangulate Each Square

Each square becomes two triangles.

There are two diagonal choices:

- lower-left to upper-right
- upper-left to lower-right

Choose the diagonal using a consistency rule, for example:

- pick the diagonal with smaller height mismatch
- or smaller plane residual
- or smaller normal discontinuity

This reduces visible folding artifacts.

A square should produce triangles only if:

- all required vertices are valid
- triangle edges are not too long
- local slope is within acceptable ground limits
- triangle residual to supporting points is acceptable

### Step 7. Ground / Non-Ground Decision

Use the triangle mesh as the ground model.

A point is ground if:

- its vertical distance to the local supporting triangle is small enough
- the triangle slope is within acceptable limits
- the triangle confidence is high enough

This is stronger than classifying at the cell level because the decision now comes from a continuous local surface rather than isolated cell centers.

## Why Triangles Are Better Than Independent Squares

Triangles do not magically make the surface continuous by themselves.

The important part is not "triangle" versus "square".
The important part is:

- shared vertices
- one common surface definition across cell borders

If you keep independent per-cell triangles, you can still have seams.
If you use shared vertices, even squares split into triangles become continuous.

So the right mental model is:

- bad: independent cell patches
- good: shared-vertex mesh

## Practical Design Choices

### Vertex Height Source

Best order to try:

1. prediction from touching cell planes
2. nearby raw point aggregation
3. interpolation from neighboring valid vertices

### Weighting

Useful weight factors:

- plane inlier ratio
- number of inlier points
- distance to the query vertex
- local plane flatness

### Invalid Regions

Do not force continuity through unsupported holes.

Instead:

- leave gaps where support is too weak
- fill only small holes
- avoid creating long triangles over missing data

### Temporal Stability

To reduce frame flicker, optionally add temporal filtering on vertex heights or mesh confidence.

This should be weak and confidence-aware so it does not lag behind real terrain changes.

## Expected Benefits

Compared with the current independent-cell representation, this method should give:

- smoother ground geometry
- fewer cracks between neighboring patches
- more stable normals
- better point-to-ground distance tests
- better behavior on sloped but continuous terrain

## Expected Failure Modes

The method can still fail if:

- the up-axis choice is wrong
- the ground is not well represented as a height field
- ToF noise is heavy near depth discontinuities
- holes are large enough that vertex inference becomes underconstrained
- smoothing is too strong and hides real edges

## If You Want the Simplest Incremental Path

Use this order:

1. keep the existing cell extraction and robust local plane fitting
2. add shared corner vertices
3. infer corner heights from neighboring cell planes
4. triangulate each square
5. classify points by triangle distance
6. add edge-aware smoothing only after the basic mesh works

That sequence gives the highest chance of getting useful results quickly.

## Short Recommendation

For this project, the best balance of robustness, continuity, and implementation risk is:

Shared-vertex triangle mesh on top of the existing regular grid, with vertex heights inferred from neighboring cell planes and protected by edge-aware smoothing.

This keeps the current structure, removes the main source of discontinuity, and is much safer than jumping directly to unstructured triangulation or a fully global spline model.
