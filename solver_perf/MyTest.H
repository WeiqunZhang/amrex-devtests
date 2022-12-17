#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFabUtil.H>

class MyTest
{
public:

    MyTest ();

    void solve ();

public: // make these public for cuda
    void initProbPoisson ();
    void initProbABecLaplacian ();

private:

    void readParameters ();
    void initData ();

    template <typename MF>
    void solvePoisson ();

    template <typename MF>
    void solveABecLaplacian ();

    std::array<int,AMREX_SPACEDIM> n_cells{AMREX_D_DECL(512,512,512)};
    int max_grid_size = 128;

    int prob_type = 2;  // 1. Poisson,  2. ABecLaplacian

    bool single_precision = false;

    // For MLMG solver
    int verbose = 1;
    int bottom_verbose = 0;
    int max_iter = 100;
    int max_fmg_iter = 0;
    int linop_maxorder = 2;
    bool agglomeration = true;
    bool consolidation = true;
    int max_coarsening_level = 30;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab exact;
    amrex::MultiFab rhs;
    amrex::MultiFab acoef;
    std::array<amrex::MultiFab,AMREX_SPACEDIM> bcoef;

    amrex::Real ascalar = 1.e-3;
    amrex::Real bscalar = 1.0;
};

template <typename MF>
void
MyTest::solvePoisson ()
{
    using namespace amrex;

    LPInfo info;
    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setMaxCoarseningLevel(max_coarsening_level);

    using T = typename MF::value_type;
    const T tol_rel = std::is_same<double,typename MF::value_type>::value ?
        T(1.e-10) : T(1.e-4);
    const Real tol_abs = 0.0;

    MLPoissonT<MF> mlpoisson({geom}, {grids}, {dmap}, info);

    mlpoisson.setMaxOrder(linop_maxorder);

    // This is a problem with Dirichlet BC
    mlpoisson.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet)},
                          {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet)});

    mlpoisson.setLevelBC(0, &exact);

    MLMGT<MF> mlmg(mlpoisson);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    solution.setVal(0.0);
    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);
}

template <typename MF>
void
MyTest::solveABecLaplacian ()
{
    using namespace amrex;

    LPInfo info;
    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setMaxCoarseningLevel(max_coarsening_level);

    using T = typename MF::value_type;
    const T tol_rel = std::is_same<double,typename MF::value_type>::value ?
        T(1.e-10) : T(1.e-4);
    const T tol_abs = T(0.0);

    MLABecLaplacianT<MF> mlabec({geom}, {grids}, {dmap}, info);

    mlabec.setMaxOrder(linop_maxorder);

    // This is a 3d problem with Dirichlet BC
    mlabec.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet)},
                       {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet)});

    
    mlabec.setLevelBC(0, &exact);

    mlabec.setScalars(ascalar, bscalar);
    mlabec.setACoeffs(0, acoef);
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));

    MLMGT<MF> mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    solution.setVal(0.0);
    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);
}

#endif
