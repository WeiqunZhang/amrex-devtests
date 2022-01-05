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
    m_index.define(ba, dm, 1, nghost);
    m_global_range.define(ba, dm);
    m_local_range.define(ba, dm);

    Vector<Long> ncells_allprocs(ParallelDescriptor::NProcs(), 0);
    for (int k = 0, N = ba.size(); k < N; ++k) {
        ncells_allprocs[dm[k]] += ba[k].numPts();
    }

    Vector<Long> rows(ncells_allprocs.size()+1);
    if (rows.size() > 0) { rows[0] = 0; }
    std::partial_sum(ncells_allprocs.begin(), ncells_allprocs.end(), rows.begin()+1);

    m_partition.define(std::move(rows));

    Long gid_begin = m_partition[ParallelDescriptor::MyProc()];
    const Long id_offset = gid_begin;
    for (MFIter mfi(m_index); mfi.isValid(); ++mfi) {
        Box const& vbx = mfi.validbox();
        Box const& fbx = mfi.fabbox();
        auto const& idarr = m_index.array(mfi);
        amrex::ParallelFor(fbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (vbx.contains(i,j,k)) {
                idarr(i,j,k) = vbx.index(IntVect{AMREX_D_DECL(i,j,k)}) + gid_begin;
            } else {
                idarr(i,j,k) = -1;
            }
        });
        m_global_range[mfi].first = gid_begin;
        gid_begin += vbx.numPts();
        m_global_range[mfi].second = gid_begin;

        m_local_range[mfi].first = m_global_range[mfi].first - id_offset;
        m_local_range[mfi].second = m_global_range[mfi].second - id_offset;
    }

    if (m_index.nGrowVect() != 0) {
        m_index.FillBoundary(geom.periodicity());
    }
}

const FabArray<BaseFab<Long> >&
RowIndex::index () const
{
    return m_index;
}

const AlgPartition&
RowIndex::partition () const
{
    return m_partition;
}

std::pair<Long,Long>
RowIndex::globalRange (MFIter const& mfi) const
{
    return m_global_range[mfi];
}

std::pair<Long,Long>
RowIndex::localRange (MFIter const& mfi) const
{
    return m_local_range[mfi];
}

}
