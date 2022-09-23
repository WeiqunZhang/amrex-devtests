#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int nlevs = 4;
        int n_cell = 32;

        Vector<IntVect> ref_ratio(nlevs,IntVect(2));
        Vector<Geometry> geom(nlevs);
        Vector<BoxArray> grids(nlevs);
        Vector<DistributionMapping> dmap(nlevs);

        Vector<MultiFab> mf(nlevs);

        RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(0.2,0.2,0.2)});
        Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
        Geometry::Setup(&rb, 0, is_periodic.data());
        Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
        Box domain = domain0;
        for (int ilev = 0; ilev < nlevs; ++ilev)
        {
            geom[ilev].define(domain);
            domain.refine(ref_ratio[ilev]);
        }

        domain = domain0;
        for (int ilev = 0; ilev < nlevs; ++ilev)
        {
            grids[ilev].define(domain);
            grids[ilev].maxSize(16);
            domain.grow(-n_cell/4);   // fine level cover the middle of the coarse domain
            domain.refine(ref_ratio[ilev]);
        }

        Vector<Real> rho = {1.2, 2.3, 3.4, 4.5};

        for (int ilev = 0; ilev < nlevs; ++ilev)
        {
            dmap[ilev].define(grids[ilev]);
            mf[ilev].define(grids[ilev], dmap[ilev], 2, 0);
            mf[ilev].setVal(rho[ilev], 1, 1);
        }

        Vector<Real> vol(nlevs, AMREX_D_TERM(0.2,*0.2,*0.2));
        for (int ilev = 1; ilev < nlevs; ++ilev) {
            vol[ilev] = vol[ilev-1] / (AMREX_D_TERM(2.,*2.,*2.));
        }

        Real sum_expected = rho[nlevs-1]*vol[nlevs-1];
        for (int ilev = 0; ilev < nlevs-1; ++ilev) {
            sum_expected += rho[ilev] * (vol[ilev] - vol[ilev+1]);
        }

        Real sum = amrex::volumeWeightedSum(GetVecOfConstPtrs(mf), 1,
                                            geom, ref_ratio);
        amrex::Print() << "sum is " << sum << ", expected value is "
                       << sum_expected << "\n";
    }
    amrex::Finalize();
}
