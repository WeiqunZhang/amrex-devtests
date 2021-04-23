#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <AMReX_BaseFab.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ncell = 128;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
        }
        Box box(IntVect(0),IntVect(ncell-1));

        BaseFab<Real> fabd(box);
        Array4<Real> const& ad = fabd.array();

        BaseFab<Real> fabh(box,1,The_Pinned_Arena());
        Array4<Real> const& ah = fabh.array();

        Gpu::synchronize();

        double t_d, t_h;

        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ad(i,j,k) = 1.0;
            });
            Gpu::synchronize();
            t_d = amrex::second()-ttmp;
        }

        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ah(i,j,k) = 1.0;
            });
            Gpu::synchronize();
            t_h = amrex::second()-ttmp;
        }

        amrex::Print() << std::scientific << "Device Fab setVal: " << t_d
                       << ", Pinned Fab setVal: " << t_h << std::endl;

        Gpu::synchronize();
    }
    amrex::Finalize();
}
