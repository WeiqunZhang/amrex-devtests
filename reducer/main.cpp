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

            Reducer<Real> reducer(Vector<ReduceOpType>{
                    ReduceOpType::min, ReduceOpType::max, ReduceOpType::sum,
                    ReduceOpType::sum, ReduceOpType::max});
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& box = mfi.tilebox();
                auto const& a = mf.array(mfi);
                reducer.eval(box, [=] AMREX_GPU_DEVICE (int iop, int i, int j, int k)
                { // 0 <= iop < 5
                    if (iop >= 0 && iop <= 2) { // min, max & sum
                        return a(i,j,k);
                    } else { // 1-norm & inf-norm
                        return std::abs(a(i,j,k));
                    }
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

            ReduceOps<ReduceOpMin, ReduceOpMax, ReduceOpSum, ReduceOpSum, ReduceOpMax> reduce_op;
            ReduceData<Real,Real,Real,Real,Real> reduce_data(reduce_op);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& box = mfi.tilebox();
                auto const& a = mf.array(mfi);
                reduce_op.eval(box, reduce_data,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k)
                               -> GpuTuple<Real,Real,Real,Real,Real>
                {
                    auto v = a(i,j,k);
                    return {v, v, v, std::abs(v), std::abs(v)};
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

            Gpu::Buffer result({std::numeric_limits<Real>::max(),
                                std::numeric_limits<Real>::lowest(),
                                Real(0), Real(0),
                                std::numeric_limits<Real>::lowest()});
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
                    Gpu::deviceReduceMin(dp  , v, gh);
                    Gpu::deviceReduceMax(dp+1, v, gh);
                    Gpu::deviceReduceSum(dp+2, v, gh);
                    Gpu::deviceReduceSum(dp+3, std::abs(v), gh);
                    Gpu::deviceReduceMax(dp+4, std::abs(v), gh);
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

            {
                ReduceOps<ReduceOpMin> reduce_op;
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

            {
                ReduceOps<ReduceOpMax> reduce_op;
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

            {
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

            {
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
                        auto v = std::abs(a(i,j,k));
                        return {v};
                    });
                }
                auto result = reduce_data.value(reduce_op);
            }

            {
                ReduceOps<ReduceOpMax> reduce_op;
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
                        auto v = std::abs(a(i,j,k));
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
