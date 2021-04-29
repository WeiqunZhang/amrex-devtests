#include <AMReX_AlgRow.H>
#include <AMReX_LayoutData.H>
#include <AMReX_Vector.H>
#include <AMReX_ParallelContext.H>

#include <limits>
#include <numeric>

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

    Vector<Long> ncells_allprocs(ParallelDescriptor::NProcs(), 0);
    for (int k = 0, N = ba.size(); k < N; ++k) {
        ncells_allprocs[dm[k]] += ba[k].numPts();
    }

    Vector<Long> rows(ncells_allprocs.size()+1);
    rows[0] = 0;
    std::partial_sum(ncells_allprocs.begin(), ncells_allprocs.end(), rows.begin()+1);

    m_partition.define(std::move(rows));

    Long id_begin = m_partition[ParallelDescriptor::MyProc()];
    for (MFIter mfi(m_id); mfi.isValid(); ++mfi) {
        Box const& vbx = mfi.validbox();
        Box const& fbx = mfi.fabbox();
        // parallel for to fill the data including ghost cells setting to lowest
        id_begin += vbx.numPts();
    }
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
