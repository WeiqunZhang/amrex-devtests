#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ncell = 256;
        int max_grid_size = 64;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};

        MultiFab mf(ba, dm, 1, 0);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            Array4<Real> const& a = mf.array(mfi);
            amrex::ParallelForRNG(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine)
            {
                a(i,j,k) = amrex::Random(engine)* 100. + amrex::Random(engine)*10. + amrex::Random(engine);
            });
        }

        auto const& ma = mf.const_arrays();
        auto mm = amrex::ParReduce(TypeList<ReduceOpMin,ReduceOpMax>{},
                                   TypeList<ValLocPair<Real,Long>,ValLocPair<Real,Long> >{}, mf,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            -> GpuTuple<ValLocPair<Real,Long>,ValLocPair<Real,Long> >
        {
            Long nc = ncell;
            Long index = i + nc*j + nc*nc*k;
            return { ValLocPair<Real,Long>{ma[box_no](i,j,k), index},
                     ValLocPair<Real,Long>{ma[box_no](i,j,k), index} };
        });

        auto minloc = amrex::get<0>(mm);
        auto maxloc = amrex::get<1>(mm);
        IntVect mincell = domain.atOffset(minloc.index);
        IntVect maxcell = domain.atOffset(maxloc.index);
        amrex::Print() << "Min value " << minloc.value << " at " << mincell
                       << "\nMax value " << maxloc.value << " at " << maxcell << std::endl;
        amrex::Print() << "MultiFab::min() = " << mf.min(0) << ", MultiFab::max() = "
                       << mf.max(0) << std::endl;
        IntVect minindex = mf.minIndex(0);
        IntVect maxindex = mf.maxIndex(0);
        amrex::Print() << "MultiFab::minIndex() = " << minindex
                       << ", MultiFab::maxIndex() = " << maxindex << std::endl;

        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            Array4<Real> const& a = mf.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (IntVect(i,j,k) == mincell) {
                    AMREX_DEVICE_PRINTF("minloc: %.17g\n", a(i,j,k));
                }
                if (IntVect(i,j,k) == maxcell) {
                    AMREX_DEVICE_PRINTF("maxloc: %.17g\n", a(i,j,k));
                }
                if (IntVect(i,j,k) == minindex) {
                    AMREX_DEVICE_PRINTF("MultiFab::minloc: %.17g\n", a(i,j,k));
                }
                if (IntVect(i,j,k) == maxindex) {
                    AMREX_DEVICE_PRINTF("MultiFab::maxloc: %.17g\n", a(i,j,k));
                }
            });
        }
    }
    amrex::Finalize();
}
