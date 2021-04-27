#ifndef AMREX_ALG_PARTITION_H_
#define AMREX_ALG_PARTITION_H_
#include <AMReX_Config.H>

#include <AMReX_INT.H>
#include <AMReX_Vector.H>

#include <memory>

namespace amrex {

class AlgPartition
{
public:

    Long numRows () const { return m_ref->m_row.back(); }

private:
    struct Ref
    {
        friend class AlgPartition;
        Vector<Long> m_row; // size: nprocs + 1
    };

    std::shared_ptr<Ref> m_ref;
};

}

#endif
