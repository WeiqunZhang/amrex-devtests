#ifndef AMREX_SP_MATRIX_H_
#define AMREX_SP_MATRIX_H_

#include <AMReX_AlgPartition.H>
#include <AMReX_Gpu.H>
#include <AMReX_INT.H>

namespace amrex {

template <typename T, template<typename> typename Allocator = DefaultAllocator>
class SpMatrix
{
public:
    typedef T value_type;

    template <class U>
    using Vec = PODVector<U,Allocator<U> >;

    SpMatrix (AlgPartition const& partition);

    ~SpMatrix ();

    SpMatrix (SpMatrix const&) = delete;
    SpMatrix (SpMatrix &&) = delete;
    void operator= (SpMatrix const&) = delete;
    void operator= (SpMatrix &&) = delete;

    Long numLocalRows () const { return m_row_end - m_row_begin; }
    Long numGlobalRows () const { return m_partition.numRows(); }

    void reserve (Long ndata);

    struct CSR {
        Vec<T> v;
        Vec<Long> col_index;
        Vec<Long> row_index;
    };

private:
    AlgPartition m_partition;
    Long m_row_begin, m_row_end;
    CSR m_data; // We might need two CSRs, one for local data, the other for remote data
};

template <typename T, template<typename> typename Allocator>
SpMatrix<T,Allocator>::SpMatrix (AlgPartition const& partition)
    : m_partition(partition),
      m_row_begin(ParallelDescriptor::MyProc()),
      m_row_end(ParallelDescriptor::MyProc()+1)
{}

template <typename T, template<typename> typename Allocator>
SpMatrix<T,Allocator>::~SpMatrix ()
{}

template <typename T, template<typename> typename Allocator>
void
SpMatrix<T,Allocator>::reserve (Long ndata)
{
//xxxxx
}

}

#endif
