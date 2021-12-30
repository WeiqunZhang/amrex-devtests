#pragma once

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>

// User Inputs
struct Inputs
{
  int use_new_fillpatch = 1;
  int ncomp = 10;
  int ngrow = 2;
  amrex::Real radius = 0.25;
};

// Level Data - Data on all boxes at a single level
struct LevelData
{

  // Consolidated Multi-Component MultiFab Data
  amrex::MultiFab cpu_mfabs, gpu_mfabs;

  // Inside-Out Mask
  amrex::MultiFab cpu_mask, gpu_mask;

};

// Main Class (derived from amrex::AmrCore)
class AmrCoreDerived : public amrex::AmrCore
{
  public:
    // Constructor
    AmrCoreDerived();

    // Destructor
    virtual ~AmrCoreDerived();

    // Initialize Data
    void InitData();

    // FillPatch Test
    void FillPatch ();

    // MakeNewLevelFromScratch Override
    virtual void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    // MakeNewLevelFromCoarse Override
    virtual void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    // RemakeLevel Override
    virtual void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    // ClearLevel Override
    virtual void ClearLevel (int lev) override;

    // ErrorEst Override
    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

  private:

    void ReadInputs();
    void CreateMultiFabs (const int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
    void FillPatchLevel (int lev, amrex::MultiFab& mf, amrex::MultiFab& mf_lm1, int icomp, int ncomp);
    void WritePlotfile();


    Inputs m_inputs;
    amrex::Vector<LevelData> m_levdata;
    amrex::Vector<amrex::BCRec> m_bcrecs;
};

