#ifndef AMREX_SP_MATRIX_H_
#define AMREX_SP_MATRIX_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_Gpu.H>
#include <AMReX_INT.H>
#include <AMReX_Scan.H>

#include <fstream>
#include <string>
#include <type_traits>

namespace amrex {

#ifdef AMREX_USE_GPU
using AlgInt = int; // CUB's SpMV does not support long int index
#else
using AlgInt = Long;
#endif

template <typename T, template<typename> class Allocator = DefaultAllocator>
class SpMatrix
{
public:
    using value_type = T;

    template <class U>
    using Vec = PODVector<U,Allocator<U> >;

    SpMatrix (AlgPartition const& partition, int stencil_size);

    SpMatrix (SpMatrix const&) = delete;
    SpMatrix& operator= (SpMatrix const&) = delete;

    SpMatrix (SpMatrix &&) = default;
    SpMatrix& operator= (SpMatrix &&) = default;

    void resize (int stencil_size);

    void assemble ();

    AlgPartition const& partition () const { return m_partition; }

    Long numLocalRows () const { return m_row_end - m_row_begin; }
    Long numGlobalRows () const { return m_partition.numGlobalRows(); }
    Long numLocalNonZero () const { return m_data.nnz; }

    T const* data () const { return m_data.mat.data(); }
    T      * data ()       { return m_data.mat.data(); }
    AlgInt const* columnIndex () const { return m_data.col_index.data(); }
    AlgInt      * columnIndex ()       { return m_data.col_index.data(); }
    AlgInt const* rowOffset () const { return m_data.row_offset.data(); }
    AlgInt      * rowOffset ()       { return m_data.row_offset.data(); }

    void printToFile (std::string const& file) const;

public: // for cuda

    template <typename N>
    void assemble_doit ();

private:

    struct CSR {
        Vec<T> mat;
        Vec<AlgInt> col_index;
        Vec<AlgInt> row_offset;
        Long nnz = -1;
    };

    AlgPartition m_partition;
    Long m_row_begin, m_row_end;
    CSR m_data; // We might need two CSRs, one for local data, the other for remote data
};

template <typename T, template<typename> class Allocator>
SpMatrix<T,Allocator>::SpMatrix (AlgPartition const& partition, int stencil_size)
    : m_partition(partition),
      m_row_begin(partition[ParallelDescriptor::MyProc()]),
      m_row_end(partition[ParallelDescriptor::MyProc()+1])
{
    static_assert(std::is_floating_point<T>::value, "SpMatrix is for floating point type only");
    resize(stencil_size);
}

template <typename T, template<typename> class Allocator>
void
SpMatrix<T,Allocator>::resize (int stencil_size)
{
    m_data.mat.resize(numLocalRows() * stencil_size);
    m_data.col_index.resize(numLocalRows() * stencil_size);
    m_data.row_offset.resize(numLocalRows() + 1);
#ifdef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT(m_data.mat.size() <
                        static_cast<std::size_t>(std::numeric_limits<AlgInt>::max()));
#endif
}

template <typename T, template<typename> class Allocator>
void
SpMatrix<T,Allocator>::assemble ()
{
    if (static_cast<Long>(m_data.mat.size()) <=
        static_cast<Long>(std::numeric_limits<int>::max())) {
        assemble_doit<int>();
    } else {
        assemble_doit<Long>();
    }
}

template <typename T, template<typename> class Allocator>
template <typename N>
void
SpMatrix<T,Allocator>::assemble_doit ()
{
    const Long old_size = m_data.mat.size();
    const Long nrows = numLocalRows();
    const int stencil_size = old_size / nrows;
    AMREX_ASSERT(nrows*stencil_size == static_cast<Long>(m_data.mat.size()));
    AMREX_ASSERT(m_data.mat.size() == m_data.col_index.size());
    AMREX_ASSERT(static_cast<Long>(m_data.row_offset.size()) == nrows + 1);

    T* mat = m_data.mat.data();
    AlgInt* col = m_data.col_index.data();
    AlgInt* row = m_data.row_offset.data();

    auto is_valid = [=] AMREX_GPU_DEVICE (N i) -> N
                    {
                        return (mat[i] != T(0)) && col[i] >= 0;
                    };

    Vec<N> psv(old_size);
    N* ps = psv.data();
    m_data.nnz = Scan::PrefixSum<N>(static_cast<N>(old_size), is_valid,
                                    [=] AMREX_GPU_DEVICE (N i, N const& s)
                                    {
                                        ps[i] = s;
                                        if (i%stencil_size == 0) {
                                            row[i/stencil_size] = s;
                                        } else if (i == old_size-1) {
                                            row[nrows] = s + static_cast<AlgInt>(is_valid(i));
                                        }
                                    },
                                    Scan::Type::exclusive, Scan::retSum);

    Vec<T> mattmpv(m_data.nnz);
    Vec<AlgInt> coltmpv(m_data.nnz);
    T* mattmp = mattmpv.data();
    AlgInt* coltmp = coltmpv.data();
    amrex::ParallelFor(old_size, [=] AMREX_GPU_DEVICE (Long i) noexcept
    {
        if (is_valid(i)) {
            mattmp[ps[i]] = mat[i];
            coltmp[ps[i]] = col[i];
        }
    });
    Gpu::streamSynchronize();
    m_data.mat.swap(mattmpv);
    m_data.col_index.swap(coltmpv);
}

template <typename T, template<typename> class Allocator>
void
SpMatrix<T,Allocator>::printToFile (std::string const& file) const
{
    std::ofstream ofs(file+"."+std::to_string(ParallelDescriptor::MyProc()));
    AllPrintToFile os(file);
    ofs << m_row_begin << " " << m_row_end << " " << m_data.nnz << "\n";
    for (Long i = 0, nrows = numLocalRows(); i < nrows; ++i) {
        Long nnz_row = m_data.row_offset[i+1] - m_data.row_offset[i];
        T    const* mat = m_data.mat.data()       + m_data.row_offset[i];
        AlgInt const* col = m_data.col_index.data() + m_data.row_offset[i];
        for (Long j = 0; j < nnz_row; ++j) {
            ofs << i << " " << col[j] << " " << mat[j] << "\n";
        }
    }
}

}

#endif
