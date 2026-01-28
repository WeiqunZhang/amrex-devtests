#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    int n_cell = 64;
    int max_grid_size = 32;
    {
        ParmParse pp;
        pp.query("n_cell", n_cell);
        pp.query("max_grid_size", max_grid_size);
    }

    constexpr int nreduce = 30;

    {
        BoxArray ba(Box(IntVect(0),IntVect(n_cell-1)));
        ba.maxSize(max_grid_size);
        MultiFab mf(ba, DistributionMapping{ba}, 1, 0);
        FillRandom(mf, 0, 1);
        mf.plus(Real(-0.2), 0, 1);

        int ntests = 3;

        double t0;
        for (int itest = 0; itest < ntests; ++itest)
        {
            auto tb = amrex::second();

            Reducer<Real> reducer(Vector<ReduceOpType>(nreduce, ReduceOpType::sum));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& box = mfi.tilebox();
                auto const& a = mf.array(mfi);
                reducer.eval(box, [=] AMREX_GPU_DEVICE (int, int i, int j, int k)
                {
                    return a(i,j,k);
                });
            }
            Vector<Real> result = reducer.getResults();

            auto te = amrex::second();
            t0 = te - tb;
        }

        double t1;
        for (int itest = 0; itest < ntests; ++itest)
        {
            auto tb = amrex::second();

            TypeMultiplier<ReduceOps, ReduceOpSum[nreduce]> reduce_op;
            TypeMultiplier<ReduceData, Real[nreduce]> reduce_data(reduce_op);
            using T = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& box = mfi.tilebox();
                auto const& a = mf.array(mfi);
                reduce_op.eval(box, reduce_data,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) -> T
                {
                    T r;
                    auto v = a(i,j,k);
                    constexpr_for<0,nreduce>([&] (auto idx) { amrex::get<idx>(r) = v; });
                    return r;
                });
            }
            auto result = reduce_data.value(reduce_op);

            auto te = amrex::second();
            t1 = te - tb;
        }

        double t2;
        for (int itest = 0; itest < ntests; ++itest)
        {
            auto tb = amrex::second();

            Vector<Real> init(nreduce, 0);
            Gpu::Buffer result(init.data(), nreduce);
            auto* dp = result.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& box = mfi.tilebox();
                auto const& a = mf.array(mfi);
                amrex::ParallelFor(Gpu::KernelInfo{}.setReduction(true), box,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k, Gpu::Handler const& gh)
                {
                    auto v = a(i,j,k);
                    for (int n = 0; n < nreduce; ++n) {
                        Gpu::deviceReduceSum(dp+n, v, gh);
                    }
                });
            }
            auto* hp = result.copyToHost();

            auto te = amrex::second();
            t2 = te - tb;
        }

        double t3;
        for (int itest = 0; itest < ntests; ++itest)
        {
            auto tb = amrex::second();

            for (int iop = 0; iop < nreduce; ++iop) {
                ReduceOps<ReduceOpSum> reduce_op;
                ReduceData<Real> reduce_data(reduce_op);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    Box const& box = mfi.tilebox();
                    auto const& a = mf.array(mfi);
                    reduce_op.eval(box, reduce_data,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k)
                                   -> GpuTuple<Real>
                    {
                        auto v = a(i,j,k);
                        return {v};
                    });
                }
                auto result = reduce_data.value(reduce_op);
            }

            auto te = amrex::second();
            t3 = te - tb;
        }

        amrex::Print() << "  VectorReduce Time: " << t0 << "\n"
                       << "  TupleReduce  Time: " << t1 << "\n"
                       << "  deviceReduce Time: " << t2 << "\n"
                       << "  SingleReduce Time: " << t3 << "\n";
    }

    amrex::Finalize();
}
