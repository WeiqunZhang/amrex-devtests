#include <AMReX_RowIndex.H>
#include <AMReX_LayoutData.H>
#include <AMReX_Vector.H>
#include <AMReX_ParallelContext.H>

#include <limits>
#include <numeric>

namespace amrex {

RowIndex::RowIndex (BoxArray const& ba, DistributionMapping const& dm,
                    Geometry const& geom, IntVect const& nghost)
{
    define(ba, dm, geom, nghost);
}

RowIndex::~RowIndex ()
{}

void RowIndex::define (BoxArray const& ba, DistributionMapping const& dm,
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
        auto const& idarr = m_id.array(mfi);
        amrex::ParallelFor(fbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (vbx.contains(i,j,k)) {
                idarr(i,j,k) = vbx.index(IntVect{AMREX_D_DECL(i,j,k)}) + id_begin;
            } else {
                idarr(i,j,k) = -1;
            }
        });
        id_begin += vbx.numPts();
    }

    if (m_id.nGrowVect() != 0) {
        m_id.FillBoundary(geom.periodicity());
    }
}

const FabArray<BaseFab<Long> >&
RowIndex::id () const
{
    return m_id;
}

const AlgPartition&
RowIndex::partition () const
{
    return m_partition;
}

}
