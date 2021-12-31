#ifndef AMREX_ROW_INDEX_H_
#define AMREX_ROW_INDEX_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_BaseFab.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_INT.H>
#include <AMReX_LayoutData.H>

#include <utility>

namespace amrex {

class RowIndex
{
public:
    RowIndex (BoxArray const& ba, DistributionMapping const& dm,
              Geometry const& geom, IntVect const& nghost);
    ~RowIndex ();

    void define (BoxArray const& ba, DistributionMapping const& dm,
                 Geometry const& geom, IntVect const& nghost);

    const FabArray<BaseFab<Long> >& index () const;
    const AlgPartition& partition () const;

    std::pair<Long,Long> range (MFIter const& mfi) const;

private:
    Geometry m_geom;
    FabArray<BaseFab<Long> > m_index;
    LayoutData<std::pair<Long,Long> > m_range;
    AlgPartition m_partition;
};

}

#endif
