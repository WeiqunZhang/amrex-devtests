#ifndef EM_PARTICLE_CONTAINER_H_
#define EM_PARTICLE_CONTAINER_H_

#include <AMReX_Particles.H>

struct PIdx
{
    enum {
        ux = 0,
        uy, uz, w,
        nattribs
    };
};

class EMParIter
    : public amrex::ParIter<0,0,PIdx::nattribs,0>
{
public:
    using amrex::ParIter<0,0,PIdx::nattribs,0>::ParIter;

    const std::array<RealVector, PIdx::nattribs>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    std::array<RealVector, PIdx::nattribs>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

class EMParticleContainer
    : public amrex::ParticleContainer<0, 0, PIdx::nattribs, 0>
{

public:

    EMParticleContainer (const amrex::Geometry            & a_geom,
                         const amrex::DistributionMapping & a_dmap,
                         const amrex::BoxArray            & a_ba,
                         const amrex::Real                 a_charge,
                         const amrex::Real                 a_mass);

    void InitParticles (const amrex::IntVect& a_num_particles_per_cell,
                        const amrex::Real     a_thermal_momentum_std,
                        const amrex::Real     a_thermal_momentum_mean,
                        const amrex::Real     a_density,
                        const amrex::RealBox& a_bounds);

    void GatherAndPush (const amrex::MultiFab& Ex,
                        const amrex::MultiFab& Ey,
                        const amrex::MultiFab& Ez,
                        const amrex::MultiFab& Bx,
                        const amrex::MultiFab& By,
                        const amrex::MultiFab& Bz,
                        amrex::Real      dt);
protected:

    amrex::Real m_charge;
    amrex::Real m_mass;
};

#endif
