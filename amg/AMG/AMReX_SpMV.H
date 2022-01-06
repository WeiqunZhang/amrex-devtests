#ifndef AMREX_SPMV_H_
#define AMREX_SPMV_H_
#include <AMReX_Config.H>

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

#if defined(AMREX_USE_CUDA)
#  include <cusparse.h>
//#  if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 11)
//#    include <cub/cub.cuh>
//#  endif
#elif defined(AMREX_USE_DPCPP)
#  include <oneapi/mkl/spblas.hpp>
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
    AlgInt const* AMREX_RESTRICT row = A.rowOffset();

#if defined(AMREX_USE_GPU)

    // y.setVal(0);

#if defined(AMREX_USE_CUDA)

#if 0

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

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSetStream(handle, Gpu::gpuStream());

    cudaDataType data_type = (sizeof(T) == sizeof(double)) ? CUDA_R_64F : CUDA_R_32F;
    cusparseIndexType_t index_type = (sizeof(AlgInt) == sizeof(int)) ?
        CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;

    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr, nrows, ncols, nnz, (void*)row, (void*)col, (void*)mat,
                      index_type, index_type, CUSPARSE_INDEX_BASE_ZERO, data_type);

    cusparseDnVecDescr_t x_descr;
    cusparseCreateDnVec(&x_descr, ncols, (void*)px, data_type);

    cusparseDnVecDescr_t y_descr;
    cusparseCreateDnVec(&y_descr, nrows, (void*)py, data_type);

    T alpha = T(1.0);
    T beta = T(0.0);

    std::size_t buffer_size;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_descr, x_descr,
                            &beta, y_descr, data_type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    void* pbuffer = (void*)The_Arena()->alloc(buffer_size);

    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_descr, x_descr,
                 &beta, y_descr, data_type, CUSPARSE_SPMV_ALG_DEFAULT, pbuffer);

    Gpu::streamSynchronize();

    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);
    cusparseDestroy(handle);
    The_Arena()->free(pbuffer);

#endif

#elif defined(AMREX_USE_HIP)

    rocsparse_handle handle;
    rocsparse_create_handle(&handle);
    rocsparse_set_stream(handle, Gpu::gpuStream());

    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);

    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);

    if constexpr (std::is_same<T,double>::value) {
        double alpha = 1.0;
        double beta = 0.0;
        rocsparse_dcsrmv(handle, rocsparse_operation_none, nrows, ncols, nnz, &alpha, descr,
                         mat, row, col, info, px, &beta, py);
    } else {
        float alpha = 1.0f;
        float beta = 0.0f;
        rocsparse_scsrmv(handle, rocsparse_operation_none, nrows, ncols, nnz, &alpha, descr,
                         mat, row, col, info, px, &beta, py);
    }

    rocsparse_destroy_mat_info(info);
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);

#elif defined(AMREX_USE_DPCPP)

    mkl::sparse::matrix_handle_t handle{};
    mkl::sparse::set_csr_data(handle, nrows, ncols, mkl::index_base::zero,
                              (AlgInt*)row, (AlgInt*)col, (T*)mat);
    mkl::sparse::gemv(Gpu::Device::streamQueue(), mkl::transpose::nontrans,
                      T(1), handle, px, T(0), py);

#endif

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
