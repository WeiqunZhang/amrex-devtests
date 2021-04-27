#ifndef AMREX_ALG_VECTOR_H_
#define AMREX_ALG_VECTOR_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_BaseFab.H>
#include <AMReX_INT.H>
#include <AMReX_REAL.H>

#include <memory>

namespace amrex {

template <typename T>
class AlgVector
    : public DataAllocator
{
public:
    typedef T value_type;

    // [begin,end) Note that end is exclusive
    AlgVector (Long a_row_begin, Long a_row_end, Arena* ar = nullptr);

    ~AlgVector ();

    AlgVector (AlgVector const&) = delete;
    void operator= (AlgVector const&) = delete;

private:
    AlgPartition m_partition;

    T* m_data; // use vector?
};

}

#endif
