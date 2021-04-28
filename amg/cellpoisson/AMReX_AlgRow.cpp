#include <AMReX_AlgRow.H>

namespace amrex {

AlgRow::AlgRow (BoxArray const& ba, DistributionMapping const& dm,
                Geometry const& geom, IntVect const& nghost)
{
    define(ba, dm, geom, nghost);
}

AlgRow::~AlgRow ()
{}

void AlgRow::define (BoxArray const& ba, DistributionMapping const& dm,
                     Geometry const& geom, IntVect const& nghost)
{
    AMREX_ASSERT(ba.ixType().cellCentered());
    m_geom = geom;
    m_id.define(ba, dm, 1, nghost);

}

const FabArray<BaseFab<Long> >&
AlgRow::id () const
{
    return m_id;
}

const AlgPartition&
AlgRow::partition () const
{
    return m_partition;
}

}
