
Compile with `make -j16 USE_[CUDA|DPCPP|HIP]=TRUE`, for CUDA, SYCL and HIP,
respectively.

The default size is `n_cells="512 512 512"`, which is usually appropriate
for one node.  To weak scale it to 2, 4, and 8 nodes, run with
`n_cells="1024 512 512"`, `"1024 1024 512"` and `"1024 1024 1024"`,
respectively.
