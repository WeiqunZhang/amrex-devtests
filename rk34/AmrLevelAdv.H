#ifndef _AmrLevelAdv_H_
#define _AmrLevelAdv_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_PhysBCFunct.H>

#include <memory>
#include <iostream>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

enum StateType { Phi_Type = 0,
                 NUM_STATE_TYPE };

/**
 * AmrLevel-derived class for hyperbolic conservation equations
 */
class AmrLevelAdv
    :
    public amrex::AmrLevel
{
public:

    /**
     * Default constructor.  Builds invalid object.
     */
    AmrLevelAdv ();

    /**
     * The basic constructor.
     */
    AmrLevelAdv (amrex::Amr&     papa,
                 int             lev,
                 const amrex::Geometry& level_geom,
                 const amrex::BoxArray& bl,
                 const amrex::DistributionMapping& dm,
                 amrex::Real            time);

    /**
     * The destructor.
     */
    virtual ~AmrLevelAdv () override;

    /**
     * Restart from a checkpoint file.
     */
    virtual void restart (amrex::Amr&   papa,
                          std::istream& is,
                          bool          bReadSpecial = false) override;

    /**
     * Write a checkpoint file.
     */
    virtual void checkPoint (const std::string& dir,
                             std::ostream&      os,
                             amrex::VisMF::How  how = amrex::VisMF::NFiles,
                             bool               dump_old = true) override;

    /**
     * Write a plotfile to specified directory.
     */
    virtual void writePlotFile (const std::string& dir,
                                std::ostream&      os,
                                amrex::VisMF::How  how) override;

    /**
     * Define data descriptors.
     */
    static void variableSetUp ();

    /**
     * Cleanup data descriptors at end of run.
     */
    static void variableCleanUp ();

    /**
     * Initialize grid data at problem start-up.
     */
    virtual void initData () override;

    /**
     * Initialize data on this level from another AmrLevelAdv (during regrid).
     */
    virtual void init (amrex::AmrLevel& old) override;

    /**
     * Initialize data on this level after regridding if old level did not previously exist
     */
    virtual void init () override;

    /**
     * Advance grids at this level in time.
     */
    virtual amrex::Real advance (amrex::Real time,
                                 amrex::Real dt,
                                 int  iteration,
                                 int  ncycle) override;

    /**
     * Estimate time step.
     */
    amrex::Real estTimeStep (amrex::Real dt_old);

    /**
     * Compute initial time step.
     */
    amrex::Real initialTimeStep ();

    /**
     * Compute initial `dt'.
     */
    virtual void computeInitialDt (int                   finest_level,
                                   int                   sub_cycle,
                                   amrex::Vector<int>&           n_cycle,
                                   const amrex::Vector<amrex::IntVect>& ref_ratio,
                                   amrex::Vector<amrex::Real>&          dt_level,
                                   amrex::Real                  stop_time) override;

    /**
     * Compute new `dt'.
     */
    virtual void computeNewDt (int                   finest_level,
                               int                   sub_cycle,
                               amrex::Vector<int>&           n_cycle,
                               const amrex::Vector<amrex::IntVect>& ref_ratio,
                               amrex::Vector<amrex::Real>&          dt_min,
                               amrex::Vector<amrex::Real>&          dt_level,
                               amrex::Real                  stop_time,
                               int                   post_regrid_flag) override;

    /**
     * Do work after timestep().
     */
    virtual void post_timestep (int iteration) override;

    /**
     * Do work after regrid().
     */
    virtual void post_regrid (int lbase, int new_finest) override;

    /**
     * Do work after a restart().
     */
    virtual void post_restart () override;

    /**
     * Do work after init().
     */
    virtual void post_init (amrex::Real stop_time) override;

    virtual void postCoarseTimeStep (amrex::Real time) override;

    /**
     * Error estimation for regridding.
     */
    virtual void errorEst (amrex::TagBoxArray& tb,
                           int                 clearval,
                           int                 tagval,
                           amrex::Real         time,
                           int          n_error_buf = 0, int ngrow = 0) override;

    static int  NUM_STATE;
    static int  NUM_GROW;

    static int verbose;

#if !defined(AMREX_USE_CUDA)
protected:
#endif

    // Function to read parameters from input file.
    static void read_params ();

    // inline functions implemented below
    AmrLevelAdv& getLevel (int lev);

    void avgDown ();

    void avgDown (int state_indx);
};

/**
 * Boundary condition function that does nothing.
 * See bc_nullfill.cpp for implementation.
 */
void nullfill (amrex::Box const& bx, amrex::FArrayBox& data,
               const int dcomp, const int numcomp,
               amrex::Geometry const& geom, const amrex::Real time,
               const amrex::Vector<amrex::BCRec>& bcr, const int bcomp,
               const int scomp);

/*
 * Inlines.
 */

inline
AmrLevelAdv&
AmrLevelAdv::getLevel (int lev)
{
    return *(AmrLevelAdv *) &parent->getLevel(lev);
}

#endif /*_AmrLevelAdv_H_*/
