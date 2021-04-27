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
        BaseFab<Real> fab(box);
        Array4<Real> const& a = fab.array();

        Real* dp = (Real*)The_Arena()->alloc(sizeof(Real));

        amrex::ParallelForRNG(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine)
        {
            a(i,j,k) = Random(engine);
            if (i == 0 && j == 0 && k == 0) { *dp = Real(0.); }
        });
        Gpu::synchronize();

        double t_atomicadd, t_atomicaddnoret;

        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
            {
                if (gh.isFullBlock()) {
                    Real r = Gpu::blockReduceSum<AMREX_GPU_MAX_THREADS>(a(i,j,k));
                    if (threadIdx.x == 0) { atomicAdd(dp,r); }
                }
            });
            Gpu::synchronize();
            t_atomicadd = amrex::second()-ttmp;
        }

        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
            {
                if (gh.isFullBlock()) {
                    Real r = Gpu::blockReduceSum<AMREX_GPU_MAX_THREADS>(a(i,j,k));
                    if (threadIdx.x == 0) { atomicAddNoRet(dp,r); }
                }
            });
            Gpu::synchronize();
            t_atomicaddnoret = amrex::second()-ttmp;
        }

        amrex::Print() << std::scientific << "atomicAdd time: " << t_atomicadd
                       << ", atomicAddNoRet time: " << t_atomicaddnoret << std::endl;

        Gpu::synchronize();
        The_Arena()->free(dp);
    }
    amrex::Finalize();
}
