#ifndef AMREX_ROW_INDEX_H_
#define AMREX_ROW_INDEX_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_BaseFab.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_INT.H>

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

private:
    Geometry m_geom;
    FabArray<BaseFab<Long> > m_index;
    AlgPartition m_partition;
};

}

#endif
