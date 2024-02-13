#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Math.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        const int N = 100'000'000;
        Gpu::DeviceVector<std::uint64_t> a(N);
        Gpu::DeviceVector<std::uint64_t> b(N);
        Gpu::DeviceVector<std::uint64_t> c(N);
        auto* pa = a.data();
        auto* pb = b.data();
        auto* pc = c.data();
        ParallelForRNG(N, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& eng)
        {
            auto r0 = Random_int(std::numeric_limits<std::uint32_t>::max(), eng);
            auto r1 = Random_int(std::numeric_limits<std::uint32_t>::max(), eng);
            auto r2 = Random_int(std::numeric_limits<std::uint32_t>::max(), eng);
            auto r3 = Random_int(std::numeric_limits<std::uint32_t>::max(), eng);
            pa[i] = r0*r1;
            pb[i] = r2*r3;
        });
        Gpu::streamSynchronize();

        double dt;
        for (int itest = 0; itest < 2; ++itest) {
            auto t0 = amrex::second();
            ParallelFor(N, [=] AMREX_GPU_DEVICE (int i)
            {
                pc[i] = Math::umulhi(pa[i], pb[i]);
            });
            Gpu::streamSynchronize();
            auto t1 = amrex::second();
            dt = t1 - t0;
        }

#ifdef AMREX_USE_GPU
        auto error = Reduce::Sum<int>(N, [=] AMREX_GPU_DEVICE (int i) {
            auto tmp = amrex::UInt128_t(pa[i]) * amrex::UInt128_t(pb[i]);
            auto r = std::uint64_t(tmp >> 64);
            return int(r != pc[i]);
        });
        if (error) {
            amrex::Print() << "  amrex::umulhi failed" << std::endl;
        }
#endif

        amrex::Print() << "  Time is " << dt << std::endl;
    }
    amrex::Finalize();
}
