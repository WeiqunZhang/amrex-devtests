#include <AMReX_AlgPartition.H>

namespace amrex {

AlgPartition::AlgPartition ()
    : m_ref(std::make_shared<Ref>())
{}

AlgPartition::AlgPartition (Vector<Long> const& rows)
    : m_ref(std::make_shared<Ref>(rows))
{}

AlgPartition::AlgPartition (Vector<Long>&& rows)
    : m_ref(std::make_shared<Ref>(std::move(rows)))
{}

AlgPartition::~AlgPartition ()
{}

void AlgPartition::define (Vector<Long> const& rows)
{
    m_ref->m_row = rows;
}

void AlgPartition::define (Vector<Long>&& rows)
{
    m_ref->m_row = std::move(rows);
}

}
