#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_EB2.H>

using namespace amrex;

void main_main ()
{
    Geometry geom;
    {
        Array<int,3> n_cell;
        ParmParse pp;
        pp.get("n_cell", n_cell);
        Array<Real,3> prob_lo, prob_hi;
        pp.get("prob_lo", prob_lo);
        pp.get("prob_hi", prob_hi);
        geom.define(Box(IntVect(0), IntVect(n_cell)-1),
                    RealBox(prob_lo, prob_hi),
                    0, {0,0,0});
    }

    EB2::Build(geom, 0, 30);

    BoxArray ba(geom.Domain());
    {
        ba.convert(IntVect(1));
        ba.maxSize(128);
    }
    DistributionMapping dm{ba};
    auto fact = amrex::makeEBFabFactory(geom, ba, dm, {2,2,2}, EBSupport::full);

    MultiFab rhs(ba, dm, 1, 0, MFInfo{}, *fact);
    MultiFab phi(ba, dm, 1, 1, MFInfo{}, *fact);
    rhs.setVal(0);
    phi.setVal(0);

    {
        int znhi = geom.Domain().bigEnd(2)+1;
        auto const& pa = phi.arrays();
        ParallelFor(phi, IntVect(1), [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            if (k == znhi) {
                pa[b](i,j,k) = -40e3;
            }
        });
    }

    {
    //    auto const& volfrac = fact->getVolFrac();
    //    VisMF::Write(volfrac, "volfrac");
    }

    LPInfo lpinfo{};
    lpinfo.setMaxCoarseningLevel(10);

    MLEBNodeFDLaplacian linop({geom}, {ba}, {dm}, lpinfo, {fact.get()});
    linop.setSigma({1.0,1.0,1.0});

    linop.setEBDirichlet([] (Real, Real , Real z) {
        if (z > 16e-3) {
            return -40e3;
        } else if (z > 9e-3) {
            return -41e3;
        } else {
            return 0.;
        }
    });

    linop.setDomainBC({LinOpBCType::Neumann,
                       LinOpBCType::Neumann,
                       LinOpBCType::Dirichlet},
                      {LinOpBCType::Neumann,
                       LinOpBCType::Neumann,
                       LinOpBCType::Dirichlet});

    MLMG mlmg(linop);
    int verbose = 2;
    int nsweeps = 2;
    int nsweeps_bottom = 8;
    {
        ParmParse pp;
        pp.query("verbose", verbose);
        pp.query("nsweeps", nsweeps);
        pp.query("nsweeps_bottom", nsweeps_bottom);
    }
    mlmg.setVerbose(verbose);
    // mlmg.setBottomSolver(BottomSolver::smoother);
    mlmg.setBottomSmooth(nsweeps_bottom);
    mlmg.setFinalSmooth(nsweeps_bottom);
    mlmg.setPreSmooth(nsweeps);
    mlmg.setPostSmooth(nsweeps);
    mlmg.solve({&phi}, {&rhs}, 1.e-6, 0);

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        main_main();
    }
    amrex::Finalize();
}
