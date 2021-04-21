#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Scan.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        unsigned int max_size = 200'000'000u;
        {
            ParmParse pp;
            int tmp = -1;
            pp.query("max_size", tmp);
            if (tmp > 0) { max_size = tmp; }
        }

        Gpu::DeviceVector<int> iv_in(max_size), iv_ou(max_size);
        Gpu::DeviceVector<Long> lv_in(max_size), lv_ou(max_size);

        int* dpi_in = iv_in.data();
        int* dpi_ou = iv_ou.data();
        Long* dpl_in = lv_in.data();
        Long* dpl_ou = lv_ou.data();

        ParallelForRNG(max_size, [=] AMREX_GPU_DEVICE (unsigned i, RandomEngine const& engine)
        {
            dpi_in[i] = static_cast<int>((Random(engine)-0.5)*100.);
            dpl_in[i] = static_cast<Long>((Random(engine)-0.5)*100.);
        });
        Gpu::synchronize();

        double t_iv_amrex, t_iv_vendor, t_lv_amrex, t_lv_vendor;
        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            Scan::PrefixSum<int>(max_size,
                                 [=] AMREX_GPU_DEVICE (unsigned i) { return dpi_in[i]; },
                                 [=] AMREX_GPU_DEVICE (unsigned i, int ps) { dpi_ou[i] = ps; },
                                 Scan::Type::exclusive, Scan::noRetSum);
            t_iv_amrex = amrex::second()-ttmp;
        }
        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            Scan::ExclusiveSum(max_size, dpi_in, dpi_ou, Scan::noRetSum);
            t_iv_vendor = amrex::second()-ttmp;
        }
        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            Scan::PrefixSum<Long>(max_size,
                                  [=] AMREX_GPU_DEVICE (unsigned i) { return dpl_in[i]; },
                                  [=] AMREX_GPU_DEVICE (unsigned i, Long ps) { dpl_ou[i] = ps; },
                                 Scan::Type::exclusive, Scan::noRetSum);
            t_lv_amrex = amrex::second()-ttmp;
        }
        for (int i = 0; i < 2; ++i) {
            double ttmp = amrex::second();
            Scan::ExclusiveSum(max_size, dpl_in, dpl_ou, Scan::noRetSum);
            t_lv_vendor = amrex::second()-ttmp;
        }

        amrex::Print() << std::scientific << "int: amrex & vendenor "
                       << t_iv_amrex << " " << t_iv_vendor << "\n"
                       << "Long: amrex & vendenor "
                       << t_lv_amrex << " " << t_lv_vendor << std::endl;
    }
    amrex::Finalize();
}
