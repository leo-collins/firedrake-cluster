Vom construction, redundant=False, on branch leo/distributed-rtree-logs
This is based on top of distributed-rtree-refactor

CSV fields:
- nprocs
- pbs_job_id
- mesh_n: Number used in UnitCubeMesh(mesh_n, mesh_n, mesh_n)
- total_mesh_cells: global number of mesh cells
- mesh_cells_per_rank
- points_per_rank
- total_points: global number of points, equal to points_per_rank*nprocs

- vom_time_s: MPI-maximum time to construct the VertexOnlyMesh. Statistics calculated over the 10 iterations in the script.
- local_time_s: distribution of construction times from each individual rank. Taken from the last iteration of the loop. This stat would show time imbalances across all ranks.

- candidate_roots: input roots on each rank. Equal to points_per_rank
- candidate_leaves: number of leaves in candidate_sf on each rank. This is how many points each rank does local point location with.
- candidate_input_peers: number of distinct input ranks amongst a destination rank's leaves. If this equals nprocs, then we are basically doing all-to-all communication instead of sparse communication.
- root_fanout: average number of leaves per root. This is 'how many replicated points' on average.

- partition_boxes: number ofg bounding boxes contributed to the partition rtree per rank
- partition_box_volume_sum: this is the total volume of a rank's bounding boxes as sued by the bounding box heuristic.

- parent_owned_cells: number of cells owned by each rank in the parent mesh.
- parent_halo_cells: number of halo cells visible to each rank

- vom_owned_points: located vom points owned by each rank after completion of the algorithm
- vom_halo_points: number of 'halo points' on each rank. These are additional points retained on each rank. I don't think these are necessary and want to make 'exclude_halos=True' the default soon.
