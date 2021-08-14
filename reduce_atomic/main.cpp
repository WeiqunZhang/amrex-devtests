#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void test_reduce_sum (Gpu::DeviceVector<float> const& dv)
{
    double t1, t2;

    {
        auto sum = Reduce::Sum(dv.size(), dv.data());
        amrex::Print() << "    Reduce::Sum          = " << sum << std::endl;
    }
    {
        double t0 = amrex::second();
        auto sum = Reduce::Sum(dv.size(), dv.data());
        t1 = amrex::second()-t0;
    }

    Gpu::PinnedVector<float> hsum(1,0.0f);
    Gpu::DeviceVector<float> dsum(1,0.0f);
    float * sp = dsum.data();
    float const* dp = dv.data();
    {
        amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), dv.size(),
        [=] AMREX_GPU_DEVICE (int i, Gpu::Handler const& h) noexcept
        {
            Gpu::deviceReduceSum(sp, dp[i], h);
//            if (threadIdx.x == 0) {
//                AMREX_DEVICE_PRINTF("xxxxx *hp = %f, *dp = %f\n", *hp, dp[i]);
//            }
        });
        Gpu::copy(Gpu::deviceToHost, dsum.begin(), dsum.end(), hsum.begin());
        Gpu::synchronize();
        amrex::Print() << "    Gpu::deviceReduceSum = " << hsum[0] << std::endl;
    }
    {
        double t0 = amrex::second();
        amrex::ParallelFor(Gpu::KernelInfo().setReduction(true), dv.size(),
        [=] AMREX_GPU_DEVICE (int i, Gpu::Handler const& h) noexcept
        {
            Gpu::deviceReduceSum(sp, dp[i], h);
        });
        Gpu::synchronize();
        t2 = amrex::second()-t0;
    }

    amrex::Print() << "    Kernel run time is " << std::scientific << t1 << " " << t2 << std::endl;
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int n = 100000000;
        {
            ParmParse pp;
            pp.query("n", n);
        }

        Gpu::DeviceVector<float> dv(n,1.0f);
        test_reduce_sum(dv);
    }
    amrex::Finalize();
}
