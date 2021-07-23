#include <main.H>
#include <AMReX_ParmParse.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>

#ifdef AMREX_USE_GPU
#define CopyHtoD amrex::Gpu::htod_memcpy
#define CopyDtoH amrex::Gpu::dtoh_memcpy
#else
#define CopyHtoD std::memcpy
#define CopyDtoH std::memcpy
#endif


// Constructor
AmrCoreDerived :: AmrCoreDerived()
{
  ReadInputs();
  m_levdata.resize(max_level+1);
  m_bcrecs.resize(1);

}

// Destructor
AmrCoreDerived :: ~AmrCoreDerived()
{
}

void AmrCoreDerived :: ReadInputs()
{
  // Read sphere radius from input file
  {
    amrex::ParmParse pp("sphere");
    pp.query("radius", m_inputs.radius);
  }

  // Read multifab params
  {
    amrex::ParmParse pp("mfab");
    pp.query("ncomp", m_inputs.ncomp);
    pp.query("ngrow", m_inputs.ngrow);
  }
}

// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization.
// overrides the pure virtual function in AmrCore
void AmrCoreDerived :: MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
  CreateMultiFabs(lev, ba, dm);
}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void AmrCoreDerived :: MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
  //amrex::Print() << "------ From MakeNewLevelFromCoarse ------" << std::endl;
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void AmrCoreDerived :: RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
}

void AmrCoreDerived :: ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
  const auto dx      = geom[lev].CellSizeArray();
  const auto prob_lo = geom[lev].ProbLoArray();
  const auto prob_hi = geom[lev].ProbHiArray();

  const amrex::MultiFab& mf = m_levdata[lev].gpu_mfabs;
  const auto tagset   = amrex::TagBox::SET;

  for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
  {
    const amrex::Box& box = mfi.validbox();

    // Box min & max
    const auto lo = box.loVect();
    const auto hi = box.hiVect();

    auto tagarray = tags.array(mfi);

    amrex::Real xc = 0.5 * ( prob_lo[0] + prob_hi[0] );
    amrex::Real yc = 0.5 * ( prob_lo[1] + prob_hi[1] );
    amrex::Real zc = 0.5 * ( prob_lo[2] + prob_hi[2] );
    amrex::Real r  = m_inputs.radius;
    amrex::Real r2 = r*r;

    amrex::ParallelFor(
      box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      amrex::Real x = prob_lo[0] + (i+0.5)* dx[0] - xc; 
      amrex::Real y = prob_lo[1] + (j+0.5)* dx[1] - yc;
      amrex::Real z = prob_lo[1] + (k+0.5)* dx[2] - zc;
      if(x*x+y*y+z*z > r2 && x*x+y*y+z*z < 1.1*r2)
        tagarray(i,j,k) = tagset;
    });

  } // end mfiter loop

}

// Delete level data
// overrides the pure virtual function in AmrCore
void AmrCoreDerived :: ClearLevel (int lev)
{
}

void AmrCoreDerived :: FillPatchLevel (int lev, amrex::MultiFab& mf, amrex::MultiFab& mf_lm1, int icomp, int ncomp)
{

  amrex::Real time = 0.0;

  if (lev == 0)
  {
    amrex::Vector<amrex::MultiFab*> smf;
    amrex::Vector<amrex::Real> stime;
    smf.push_back(&mf);
    stime.push_back(0.0);

    if(amrex::Gpu::inLaunchRegion())
    {
        amrex::PhysBCFunctNoOp physbc;
        amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp, geom[lev], physbc, 0);
    }
    else
    {
      amrex::PhysBCFunctNoOp physbc;
      amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                  geom[lev], physbc, 0);
    }
  }
  else
  {
    amrex::Vector<amrex::MultiFab*> cmf, fmf;
    amrex::Vector<amrex::Real> ctime, ftime;
    cmf.push_back(&mf_lm1);
    ctime.push_back(0.0);
    fmf.push_back(&mf);
    ftime.push_back(0.0);

    amrex::Interpolater* mapper = &amrex::cell_cons_interp;
    //amrex::Interpolater* mapper = &amrex::pc_interp;

    if(amrex::Gpu::inLaunchRegion())
    {
      amrex::PhysBCFunctNoOp cphysbc;
      amrex::PhysBCFunctNoOp fphysbc;

      amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                0, icomp, ncomp, geom[lev-1], geom[lev],
                                cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                mapper, m_bcrecs, 0);
    }
    else
    {
      amrex::PhysBCFunctNoOp cphysbc;
      amrex::PhysBCFunctNoOp fphysbc;

      amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                0, icomp, ncomp, geom[lev-1], geom[lev],
                                cphysbc, 0, fphysbc, 0,
                                refRatio(lev-1), mapper, m_bcrecs, 0);
    }
  }
}

void AmrCoreDerived :: InitData()
{
  const amrex::Real time = 0.0;

  InitFromScratch(time);

  printGridSummary(amrex::OutStream(),0,finest_level);

  WritePlotfile();

}

void AmrCoreDerived :: FillPatch()
{
  const int icomp = 0;

  for(int lev = 0; lev <= finest_level; ++lev)
    FillPatchLevel(lev, m_levdata[lev].gpu_mfabs, m_levdata[lev-1].gpu_mfabs, icomp, m_levdata[lev].gpu_mfabs.nComp() );

}

void AmrCoreDerived :: CreateMultiFabs(const int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{

  const auto ncomp = m_inputs.ncomp;
  const auto ngrow = m_inputs.ngrow;

  // Single component multifab for mask (0/1)
  //m_levdata[lev].cpu_mask.define(ba, dm, 1, ngrow, amrex::MFInfo().SetArena(amrex::The_Cpu_Arena()));

  // Allocate CPU memory
  m_levdata[lev].cpu_mfabs.define(ba, dm, ncomp, ngrow, amrex::MFInfo().SetArena(amrex::The_Cpu_Arena()));

  // Allocate GPU memory
  m_levdata[lev].gpu_mfabs.define(ba, dm, ncomp, ngrow, amrex::MFInfo().SetArena(amrex::The_Arena()));

  m_levdata[lev].gpu_mfabs.setVal(1);
}

void AmrCoreDerived :: WritePlotfile()
{
  // Write Plotfile to disk
  {
    const auto ncomp = m_inputs.ncomp;

    // Copy mfab data from gpu to cpu
    for (int lev = 0; lev <= finest_level; ++lev)
      amrex::dtoh_memcpy(m_levdata[lev].cpu_mfabs, m_levdata[lev].gpu_mfabs);

    // Set current time
    amrex::Real curr_time = 0.0;

    // Set current step
    amrex::Real curr_step = 0.0;

    // Initialize current step vector at all refinement levels with 0
    amrex::Vector<int> step_vec(finest_level+1, curr_step);

    // Multifab component names
    amrex::Vector<std::string> var_names;
    var_names.resize(ncomp);
    for(int icomp = 0; icomp < ncomp; ++icomp)
      var_names[icomp] = "v" + std::to_string(icomp+1);

    // Vector of Output Multifabs for output to disk (vec size = num levels)
    amrex::Vector<amrex::MultiFab> mf_out(finest_level + 1);

    // Copy data from CPU MultiFabs to Output MultiFabs
    for (int lev = 0; lev <= finest_level; ++lev)
    {
      mf_out[lev].define(grids[lev], dmap[lev], ncomp, 0, amrex::MFInfo().SetArena(amrex::The_Cpu_Arena()));

      for(int icomp = 0; icomp < ncomp; ++icomp)
      {
        amrex::Gpu::LaunchSafeGuard lsg(false);
        amrex::MultiFab mf_icomp(m_levdata[lev].cpu_mfabs, amrex::make_alias, icomp, 1);
        amrex::MultiFab::Copy(mf_out[lev], mf_icomp, 0, icomp, 1, 0);
      }

    }

    // Plotfile name
    const std::string& plot_file_name = "plt";

    amrex::WriteMultiLevelPlotfile(plot_file_name, finest_level+1, amrex::GetVecOfConstPtrs(mf_out), var_names, Geom(), curr_time, step_vec, refRatio());

  }

}
