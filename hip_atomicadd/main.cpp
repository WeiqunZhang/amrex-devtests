#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <AMReX_BaseFab.H>

using namespace amrex;

//#define ATOMIC_ADD_SHARED

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
        Gpu::streamSynchronize();

        double t_atomicadd, t_atomicaddnoret, t_unsafeatomicadd, t_amrexatomicadd;

        for (int i = 0; i < 2; ++i) {
            amrex::single_task([=] AMREX_GPU_DEVICE () { *dp = 0._rt; });
            Gpu::streamSynchronize();
            double ttmp = amrex::second();
            amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
            {
#ifdef ATOMIC_ADD_SHARED
                __shared__ Real r;
                if (threadIdx.x == 0) { r = 0._rt; }
                __syncthreads();
                atomicAdd(&r, a(i,j,k));
                __syncthreads();
                if (threadIdx.x == 0) { atomicAdd(dp,r); }
#else
                if (gh.isFullBlock()) {
                    Real r = Gpu::blockReduceSum<AMREX_GPU_MAX_THREADS>(a(i,j,k));
                    if (threadIdx.x == 0) { atomicAdd(dp,r); }
                }
#endif
            });
            Gpu::streamSynchronize();
            t_atomicadd = amrex::second()-ttmp;
            Real result;
            Gpu::dtoh_memcpy(&result, dp, sizeof(Real));
            amrex::Print() << "atomicAdd Result is " << result << std::endl;
        }

        for (int i = 0; i < 2; ++i) {
            amrex::single_task([=] AMREX_GPU_DEVICE () { *dp = 0._rt; });
            Gpu::streamSynchronize();
            double ttmp = amrex::second();
            amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
            {
#ifdef ATOMIC_ADD_SHARED
                __shared__ Real r;
                if (threadIdx.x == 0) { r = 0._rt; }
                __syncthreads();
                atomicAddNoRet(&r, a(i,j,k));
                __syncthreads();
                if (threadIdx.x == 0) { atomicAddNoRet(dp,r); }
#else
                if (gh.isFullBlock()) {
                    Real r = Gpu::blockReduceSum<AMREX_GPU_MAX_THREADS>(a(i,j,k));
                    if (threadIdx.x == 0) { atomicAddNoRet(dp,r); }
                }
#endif
            });
            Gpu::streamSynchronize();
            t_atomicaddnoret = amrex::second()-ttmp;
            Real result;
            Gpu::dtoh_memcpy(&result, dp, sizeof(Real));
            amrex::Print() << "atomicAddNoRet result is " << result << std::endl;
        }

        for (int i = 0; i < 2; ++i) {
            amrex::single_task([=] AMREX_GPU_DEVICE () { *dp = 0._rt; });
            Gpu::streamSynchronize();
            double ttmp = amrex::second();
            amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
            {
#ifdef ATOMIC_ADD_SHARED
                __shared__ Real r;
                if (threadIdx.x == 0) { r = 0._rt; }
                __syncthreads();
                unsafeAtomicAdd(&r, a(i,j,k));
                __syncthreads();
                if (threadIdx.x == 0) { unsafeAtomicAdd(dp,r); }
#else
                if (gh.isFullBlock()) {
                    Real r = Gpu::blockReduceSum<AMREX_GPU_MAX_THREADS>(a(i,j,k));
                    if (threadIdx.x == 0) { unsafeAtomicAdd(dp,r); }
                }
#endif
            });
            Gpu::streamSynchronize();
            t_unsafeatomicadd = amrex::second()-ttmp;
            Real result;
            Gpu::dtoh_memcpy(&result, dp, sizeof(Real));
            amrex::Print() << "unsafeAtomicAdd result is " << result << std::endl;
        }

        for (int i = 0; i < 2; ++i) {
            amrex::single_task([=] AMREX_GPU_DEVICE () { *dp = 0._rt; });
            Gpu::streamSynchronize();
            double ttmp = amrex::second();
            amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
            {
#ifdef ATOMIC_ADD_SHARED
                __shared__ Real r;
                if (threadIdx.x == 0) { r = 0._rt; }
                __syncthreads();
                Gpu::Atomic::Add(&r, a(i,j,k));
                __syncthreads();
                if (threadIdx.x == 0) { Gpu::Atomic::Add(dp,r); }
#else
                if (gh.isFullBlock()) {
                    Real r = Gpu::blockReduceSum<AMREX_GPU_MAX_THREADS>(a(i,j,k));
                    if (threadIdx.x == 0) { Gpu::Atomic::Add(dp,r); }
                }
#endif
            });
            Gpu::streamSynchronize();
            t_amrexatomicadd = amrex::second()-ttmp;
            Real result;
            Gpu::dtoh_memcpy(&result, dp, sizeof(Real));
            amrex::Print() << "amrexatomicadd result is " << result << std::endl;
        }

        amrex::Print() << std::scientific << "atomicAdd time: " << t_atomicadd
                       << ", atomicAddNoRet time: " << t_atomicaddnoret
                       << ", unsafeAtomicAdd time: " << t_unsafeatomicadd
                       << ", amrexatomicadd time: " << t_amrexatomicadd << std::endl;

        Gpu::streamSynchronize();
        The_Arena()->free(dp);
    }
    amrex::Finalize();
}
