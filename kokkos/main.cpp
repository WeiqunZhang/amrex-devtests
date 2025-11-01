#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <Kokkos_Core.hpp>

using namespace amrex;

static void test_amrex (MultiFab& mfa, MultiFab const& mfb)
{
    for (MFIter mfi(mfa); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        Array4<Real const> const& b = mfb.const_array(mfi);
        Array4<Real> const& a = mfa.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            a(i,j,k) += 0.5*b(i,j,k);
        });
    }
}

static void test_kokkos (MultiFab& mfa, MultiFab const& mfb)
{
    for (MFIter mfi(mfa); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto N0 = bx.length(0);
        auto N1 = bx.length(1);
        auto N2 = bx.length(2);
        auto* pa = mfa[mfi].dataPtr();
        auto const* pb = mfb[mfi].dataPtr();
        Kokkos::View<double***,Kokkos::LayoutLeft> a(pa,N0,N1,N2);
        Kokkos::View<double const***, Kokkos::LayoutLeft> b(pb,N0,N1,N2);
        Kokkos::parallel_for(Kokkos::MDRangePolicy({0,0,0},{N0,N1,N2}),
                             KOKKOS_LAMBDA (int i, int j, int k)
        {
            a(i,j,k) += 0.5*b(i,j,k);
        });
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(amrex::Gpu::Device::deviceId()));
    {
        BoxArray ba(Box(IntVect(0),IntVect(257)));
        DistributionMapping dm(ba);
        MultiFab mfa(ba, dm, 1, 0);
        MultiFab mfb(ba, dm, 1, 0);
        mfa.setVal(1.0);
        mfb.setVal(2.0);
        double tamrex = 1.e10;
        double tkokkos = 1.e10;
        for (int count = 0; count < 1; ++count) {
            amrex::Gpu::synchronize();
            double t0 = amrex::second();

            test_amrex(mfa, mfb);

            amrex::Gpu::synchronize();
            double t1 = amrex::second();

            test_kokkos(mfa, mfb);

            amrex::Gpu::synchronize();
            double t2 = amrex::second();

            tamrex = std::min(t1-t0, tamrex);
            tkokkos = std::min(t2-t1, tkokkos);
        }
        std::cout << "amrex  run time is " << tamrex << "\n"
                  << "kokkos run time is " << tkokkos << "\n";
    }
    Kokkos::finalize();
    amrex::Finalize();
}
