#ifndef AMREX_SPMV_H_
#define AMREX_SPMV_H_
#include <AMReX_Config.H>

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

#if defined(AMREX_USE_CUDA) && defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 11)
#  include <cub/cub.cuh>
#endif

namespace amrex {

template <typename T>
void SpMV (AlgVector<T>& y, SpMatrix<T> const& A, AlgVector<T> const& x)
{
    AMREX_ASSERT(x.numLocalRows() == y.numLocalRows());
    AMREX_ASSERT(x.numGlobalRows() == y.numGlobalRows());

    Long const nrows = A.numLocalRows();
    Long const ncols = y.numLocalRows();
    Long const nnz = A.numLocalNonZero();

    T      * AMREX_RESTRICT py = y.data();
    T const* AMREX_RESTRICT px = x.data();
    T const* AMREX_RESTRICT mat = A.data();
    AlgInt const* AMREX_RESTRICT col = A.columnIndex();
    AlgInt const* AMREX_RESTRICT row = A.rowIndex();

#ifdef AMREX_USE_GPU

#if defined(AMREX_USE_CUDA)

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 11)
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, (T*)mat, (AlgInt*)row, (AlgInt*)col,
                           (T*)px, (T*)py, nrows, ncols, nnz, Gpu::gpuStream());
    d_temp_storage = (void*)The_Arena()->alloc(temp_storage_bytes);
    cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, (T*)mat, (AlgInt*)row, (AlgInt*)col,
                           (T*)px, (T*)py, nrows, ncols, nnz, Gpu::gpuStream());
    Gpu::streamSynchronize();
    The_Arena()->free(d_temp_storage);

#else
    amrex::Abort("SpMV needs CUDA >= 11");
#endif

#elif defined(AMREX_USE_HIP)

    amrex::Abort("xxxxx HIP todo: SpMV");

#elif defined(AMREX_USE_DPCPP)

    amrex::Abort("xxxxx DPCPP todo: SpMV");

#endif

    Gpu::streamSynchronize();
    AMREX_GPU_ERROR_CHECK();

#else

    for (Long i = 0; i < ncols; ++i) {
        T r = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel for reduction(+:r)
#endif
        for (Long j = row[i]; j < row[i+1]; ++j) {
            r += mat[j] * px[col[j]];
        }
        py[i] = r;
    }

#endif
}

}

#endif
