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
    AlgPartition ();
    AlgPartition (Vector<Long> const& rows);
    AlgPartition (Vector<Long>&& rows);
    ~AlgPartition ();

    void define (Vector<Long> const& rows);
    void define (Vector<Long>&& rows);

    Long operator[] (int i) const { return m_ref->m_row[i]; }
    Long numGlobalRows () const { return m_ref->m_row.back(); }

private:
    struct Ref
    {
        friend class AlgPartition;
        Ref () = default;
        Ref (Vector<Long> const& rows) : m_row(rows) {}
        Ref (Vector<Long>&& rows) : m_row(std::move(rows)) {}
        Vector<Long> m_row; // size: nprocs + 1
    };

    std::shared_ptr<Ref> m_ref;
};

}

#endif
