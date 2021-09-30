#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void test_reduceops (iMultiFab const& mf)
{
    double tnew, told;
    int rnew, rold;
    for (int itest = 0; itest < 2; ++itest) {
        double t0 = amrex::second();
        {
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<int> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            auto const& ma = mf.const_arrays();

            reduce_op.eval(mf, IntVect(0), reduce_data,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept -> ReduceTuple
            {
                return { ma[box_no](i, j, k) };
            });

            ReduceTuple hv = reduce_data.value(reduce_op);
            rnew = amrex::get<0>(hv);
        }
        double t1 = amrex::second();
        tnew = t1-t0;
        {
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<int> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (MFIter mfi(mf); mfi.isValid(); ++mfi)
            {
                Box const& bx = mfi.validbox();
                auto const& fab = mf.const_array(mfi);
                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
                {
                    return { fab(i, j, k) };
                });
            }

            ReduceTuple hv = reduce_data.value(reduce_op);
            rold = amrex::get<0>(hv);
        }
        double t2 = amrex::second();
        told = t2-t1;
    }

    AMREX_ALWAYS_ASSERT(rnew == mf.boxArray().numPts() && rnew == rold);

    amrex::Print() << "    Kernel run times are " << std::scientific << told
                   << " " << tnew << ".\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int ncell = 256;
        int max_grid_size;
        std::vector<int> box_sizes;
        std::vector<int> nboxes;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.query("max_grid_size", max_grid_size);
            pp.queryarr("box_sizes", box_sizes);
            nboxes.resize(box_sizes.size(),1);
            pp.queryarr("nboxes", nboxes);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba;
        if (box_sizes.empty()) {
            ba = BoxArray(domain);
            ba.maxSize(max_grid_size);
        } else {
            BoxList bl;
            for (int i = 0; i < box_sizes.size(); ++i) {
                for (int j = 0; j < nboxes[i]; ++j) {
                    bl.push_back(Box(IntVect(0), IntVect(box_sizes[i]-1)));
                }
            }
            ba = BoxArray(std::move(bl));
        }
        DistributionMapping dm{ba};
        iMultiFab mf(ba, dm, 1, 0);
        mf.setVal(1);

        test_reduceops(mf);
    }
    amrex::Finalize();
}
