#ifndef AMREX_ALG_ROW_H_
#define AMREX_ALG_ROW_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_BaseFab.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_INT.H>

namespace amrex {

class AlgRow
{
public:
    AlgRow (BoxArray const& ba, DistributionMapping const& dm,
            Geometry const& geom, IntVect const& nghost);
    ~AlgRow ();

    void define (BoxArray const& ba, DistributionMapping const& dm,
                 Geometry const& geom, IntVect const& nghost);

    const FabArray<BaseFab<Long> >& id () const;
    const AlgPartition& partition () const;

private:
    Geometry m_geom;
    FabArray<BaseFab<Long> > m_id;
    AlgPartition m_partition;
};

}

#endif
