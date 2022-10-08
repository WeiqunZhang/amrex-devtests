#include <AMReX_TagBox.H>
#include <AMReX_ParmParse.H>

#include "AmrLevelAdv.H"
#include "Prob.H"

using namespace amrex;

int      AmrLevelAdv::verbose = 0;
int      AmrLevelAdv::NUM_STATE       = 1;  // One variable in the state
int      AmrLevelAdv::NUM_GROW        = 1;  // number of ghost cells

/**
 * Default constructor.  Builds invalid object.
 */
AmrLevelAdv::AmrLevelAdv ()
{
}

/**
 * The basic constructor.
 */
AmrLevelAdv::AmrLevelAdv (Amr&            papa,
                          int             lev,
                          const Geometry& level_geom,
                          const BoxArray& bl,
                          const DistributionMapping& dm,
                          Real            time)
    :
    AmrLevel(papa,lev,level_geom,bl,dm,time)
{
}

/**
 * The destructor.
 */
AmrLevelAdv::~AmrLevelAdv ()
{
}

/**
 * Restart from a checkpoint file.
 */
void
AmrLevelAdv::restart (Amr&          papa,
                      std::istream& is,
                      bool          bReadSpecial)
{
}

/**
 * Write a checkpoint file.
 */
void
AmrLevelAdv::checkPoint (const std::string& dir,
                         std::ostream&      os,
                         VisMF::How         how,
                         bool               dump_old)
{
}

/**
 * Write a plotfile to specified directory.
 */
void
AmrLevelAdv::writePlotFile (const std::string& dir,
                             std::ostream&      os,
                            VisMF::How         how)
{
}

/**
 * Define data descriptors.
 */
void
AmrLevelAdv::variableSetUp ()
{
    BL_ASSERT(desc_lst.size() == 0);

    // Get options, set phys_bc
    read_params();

    desc_lst.addDescriptor(Phi_Type,IndexType::TheCellType(),
                           StateDescriptor::Point,NUM_GROW,NUM_STATE,
                           &cell_cons_interp);

    int lo_bc[BL_SPACEDIM];
    int hi_bc[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; ++i) {
        lo_bc[i] = hi_bc[i] = BCType::int_dir;   // periodic boundaries
    }

    BCRec bc(lo_bc, hi_bc);

    StateDescriptor::BndryFunc bndryfunc(nullfill);
    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.

    desc_lst.setComponent(Phi_Type, 0, "phi", bc,
                          bndryfunc);
}

/**
 * Cleanup data descriptors at end of run.
 */
void
AmrLevelAdv::variableCleanUp ()
{
    desc_lst.clear();
}

/**
 * Initialize grid data at problem start-up.
 */
void
AmrLevelAdv::initData ()
{
    // Initialize data on MultiFab
    initdata(get_new_data(Phi_Type), geom);
}

/**
 * Initialize data on this level from another AmrLevelAdv (during regrid).
 */
void
AmrLevelAdv::init (AmrLevel &old)
{
    AmrLevelAdv* oldlev = (AmrLevelAdv*) &old;

    //
    // Create new grid data by fillpatching from old.
    //
    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev->state[Phi_Type].curTime();
    Real prev_time = oldlev->state[Phi_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(Phi_Type);

    FillPatch(old, S_new, 0, cur_time, Phi_Type, 0, NUM_STATE);
}

/**
 * Initialize data on this level after regridding if old level did not previously exist
 */
void
AmrLevelAdv::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[Phi_Type].curTime();
    Real prev_time = getLevel(level-1).state[Phi_Type].prevTime();

    Real dt_old = (cur_time - prev_time)/(Real)parent->MaxRefRatio(level-1);

    setTimeLevel(cur_time,dt_old,dt);
    MultiFab& S_new = get_new_data(Phi_Type);
    FillCoarsePatch(S_new, 0, cur_time, Phi_Type, 0, NUM_STATE);
}

/**
 * Advance grids at this level in time.
 */
Real
AmrLevelAdv::advance (Real time,
                      Real dt,
                      int  iteration,
                      int  ncycle)
{
    for (int k = 0; k < NUM_STATE_TYPE; k++) {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }
    amrex::Print() << "Adance Level " << level << ": dt = " << dt << std::endl;

    ParmParse pp("adv");
    int rk_order = 2;
    pp.query("rk_order", rk_order);

    RK(rk_order, Phi_Type, time, dt, iteration, ncycle,
       [&] (int stage, MultiFab& dSdt, MultiFab const& S, Real)
       {
           auto const& sa = S.const_arrays();
           auto const& sdot = dSdt.arrays();
           amrex::ParallelFor(S, [=] (int bi, int i, int j, int k)
           {
               Real a = (sa[bi](i,j,k)
                         + sa[bi](i-1,j,k) + sa[bi](i+1,j,k)
                         + sa[bi](i,j-1,k) + sa[bi](i,j+1,k))/5.;
               sdot[bi](i,j,k) = std::pow(a,0.4);
           });
       });

    return dt;
}

void
AmrLevelAdv::postCoarseTimeStep (Real time)
{
    AmrLevel::postCoarseTimeStep(time);

    for (int lev = 0; lev < 2; ++lev)
    {
        MultiFab const& S_new = getLevel(lev).get_new_data(Phi_Type);
        Real smin = S_new.min(0);
        Real smax = S_new.max(0);
        Real sexact = std::pow(1.+0.6*time, 5./3.);
        amrex::Print() << "\n    Level = " << lev << " Time = " << time
                       << " Error: [" << smin-sexact
                       << ", " << smax-sexact << "], exact = " << sexact << "\n\n";
    }
}

/**
 * Estimate time step.
 */
Real
AmrLevelAdv::estTimeStep (Real)
{
    ParmParse pp("adv");
    Real dt;
    pp.get("fixed_dt", dt);
    for (int lev = 0; lev < level; ++lev) {
        dt /= 2;
    }
    return dt;
}

/**
 * Compute initial time step.
 */
Real
AmrLevelAdv::initialTimeStep ()
{
    return estTimeStep(0.0);
}

/**
 * Compute initial `dt'.
 */
void
AmrLevelAdv::computeInitialDt (int                   finest_level,
                               int                   /*sub_cycle*/,
                               Vector<int>&           n_cycle,
                               const Vector<IntVect>& /*ref_ratio*/,
                               Vector<Real>&          dt_level,
                               Real                  stop_time)
{
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0)
        return;

    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
    }
}

/**
 * Compute new `dt'.
 */
void
AmrLevelAdv::computeNewDt (int                   finest_level,
                           int                   /*sub_cycle*/,
                           Vector<int>&           n_cycle,
                           const Vector<IntVect>& /*ref_ratio*/,
                           Vector<Real>&          dt_min,
                           Vector<Real>&          dt_level,
                           Real                  stop_time,
                           int                   post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0)
        return;

    for (int i = 0; i <= finest_level; i++)
    {
        AmrLevelAdv& adv_level = getLevel(i);
        dt_min[i] = adv_level.estTimeStep(dt_level[i]);
    }
}

/**
 * Do work after timestep().
 */
void
AmrLevelAdv::post_timestep (int iteration)
{
    AmrLevel::post_timestep(iteration);

    if (level < parent->finestLevel()) {
        avgDown();
    }
}

/**
 * Do work after regrid().
 */
void
AmrLevelAdv::post_regrid (int lbase, int /*new_finest*/) {
}

/**
 * Do work after a restart().
 */
void
AmrLevelAdv::post_restart()
{
}

/**
 * Do work after init().
 */
void
AmrLevelAdv::post_init (Real /*stop_time*/)
{
}

/**
 * Error estimation for regridding.
 */
void
AmrLevelAdv::errorEst (TagBoxArray& tags,
                       int          /*clearval*/,
                       int          /*tagval*/,
                       Real         /*time*/,
                       int          /*n_error_buf*/,
                       int          /*ngrow*/)
{
    const char tagval = TagBox::SET;
    IntVect domainsize = geom.Domain().size();
    auto const& a = tags.arrays();
    amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE
        (int bi, int i, int j, int k)
    {
        if (IntVect(AMREX_D_DECL(i,j,k))*2 == domainsize) {
            a[bi](i,j,k) = tagval;
        }
    });
}

/**
 * Read parameters from input file.
 */
void
AmrLevelAdv::read_params ()
{
}

void
AmrLevelAdv::avgDown ()
{
    if (level == parent->finestLevel()) return;
    avgDown(Phi_Type);
}

void
AmrLevelAdv::avgDown (int state_indx)
{
    if (level == parent->finestLevel()) return;

    AmrLevelAdv& fine_lev = getLevel(level+1);
    MultiFab&  S_fine   = fine_lev.get_new_data(state_indx);
    MultiFab&  S_crse   = get_new_data(state_indx);

    amrex::average_down(S_fine,S_crse,
                         fine_lev.geom,geom,
                         0,S_fine.nComp(),parent->refRatio(level));
}
