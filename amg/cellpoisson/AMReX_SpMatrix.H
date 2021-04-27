#ifndef AMREX_SP_MATRIX_H_
#define AMREX_SP_MATRIX_H_

#include <AMReX_REAL.H>
#include <AMReX_INT.H>
#include <AMReX_BaseFab.H>

#include <memory>

namespace amrex {

template <typename T>
class SpMatrix
    : public DataAllocator
{
public:
    typedef T value_type;

    // [begin,end) Note that end is exclusive
//    SpMatrix (Long a_row_begin, Long a_row_end, Arena* ar = nullptr);

//    ~SpMatrix ();

    SpMatrix (SpMatrix const&) = delete;
    void operator= (SpMatrix const&) = delete;

private:
    // distribution mapping?
    // Let's assume that the global lo is always 0
    //
    // Do we allow empty on this process?
    //
    // Something equivalent to BoxArray and DistributionMapping?

//    Long m_num_global_rows;

//    Long m_row_begin;
//    Long m_row_end;
//    Real* m_data;
};

}

#endif
