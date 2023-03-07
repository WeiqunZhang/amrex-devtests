#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int ncell = 128;
        int max_grid_size = 32;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);

        DistributionMapping dm{ba};

        const auto dx = 2.0/static_cast<Real>(ncell);

        MultiFab mf(ba, dm, 1, 0);
        auto const& a = mf.arrays();
        ParallelFor(mf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            Real z = -1.0 + k*dx;
            a[bno](i,j,k) = std::exp(-z*z);
        });

        int direction = 2; // z-direction
        using T = KeyValuePair<Real,int>;
        BaseFab<T> r = ReduceToPlane<ReduceOpMax,T>
            (direction, domain, mf,
             [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> T
             {
                 return {a[box_no](i,j,k), k};
             });

        auto const& ar = r.const_array();
        AMREX_LOOP_3D(r.box(), i, j, k,
        {
            std::cout << "   r("<<i<<","<<j<<","<<k<<") = " << ar(i,j,k).first()
                      << " " << ar(i,j,k).second() << "\n";
        });
    }
    amrex::Finalize();
}
