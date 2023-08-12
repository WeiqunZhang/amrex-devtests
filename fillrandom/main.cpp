#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void fill_random_host_api (MultiFab& mf)
{
    Real mean = 1.3;
    Real stddev = 0.2;
    amrex::FillRandomNormal(mf, 0, 1, mean, stddev);
}

void fill_random_device_api (MultiFab& mf)
{
    Real mean = 1.3;
    Real stddev = 0.2;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const& bx = mfi.fabbox();
        auto const& a = mf.array(mfi);
        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& eng)
        {
            a(i,j,k) = amrex::RandomNormal(mean, stddev, eng);
        });
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BoxArray ba(Box(IntVect(0), IntVect(255)));
        ba.maxSize(128);
        DistributionMapping dm{ba};
        MultiFab mf(ba,dm,1,1);
        mf.setVal(0.0);

        for (int i = 0; i < 4; ++i) {
            auto t0 = amrex::second();
            fill_random_host_api(mf);
            auto t1 = amrex::second();
            fill_random_device_api(mf);
            auto t2 = amrex::second();
            amrex::Print() << "#" << i << ": host_api time is " << t1-t0
                           << ", device_api time is " << t2-t1 << std::endl;
        }
    }
    amrex::Finalize();
}
