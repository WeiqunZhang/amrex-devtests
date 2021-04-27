#ifndef AMREX_SP_MATRIX_H_
#define AMREX_SP_MATRIX_H_

#include <AMReX_AlgPartition.H>
#include <AMReX_BaseFab.H>
#include <AMReX_INT.H>
#include <AMReX_REAL.H>

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
    AlgPartition m_partition;
};

}

#endif
