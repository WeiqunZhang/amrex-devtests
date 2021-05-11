#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

using namespace amrex;

template <typename MF>
void test_reduce_sum (BoxArray const& ba, DistributionMapping const& dm,
                      Box const& domain)
{
    double t, tvendor=0., tvec, t1d;

    using value_type = typename MF::value_type;

    MF mf(ba,dm,1,0);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        Array4<value_type> const& a = mf.array(mfi);
        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine)
        {
            a(i,j,k) = static_cast<value_type>(amrex::Random(engine) + 0.5_rt);
        });
    }

    {
        BL_PROFILE("reduce-warmup");
        auto r = mf.sum(0);
        if (std::is_same<MF,MultiFab>::value) {
            amrex::Print() << "    MultiFab::sum       = " << r << std::endl;
        } else {
            amrex::Print() << "    iMultiFab::sum      = " << r << std::endl;
        }
    }
    {
        BL_PROFILE("reduce-mf");
        double t0 = amrex::second();
        mf.sum(0);
        t = amrex::second()-t0;
    }

    auto p = mf[0].dataPtr();
    auto hsum = (value_type*)The_Pinned_Arena()->alloc(sizeof(value_type));
    auto dsum = (value_type*)The_Arena()->alloc(sizeof(value_type));
    Long npts = domain.numPts();

#if defined(AMREX_USE_CUB)
    for (int i = 0; i < 2; ++i) {
        double t0 = amrex::second();
        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, dsum, npts);
        // Allocate temporary storage
        d_temp_storage = (void*)The_Arena()->alloc(temp_storage_bytes);
        // Run sum-reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, dsum, npts);
        Gpu::dtoh_memcpy(hsum,dsum, sizeof(value_type));
        The_Arena()->free(d_temp_storage);
        if (i == 0) {
            amrex::Print() << "    cub sum             = " << *hsum << std::endl;
        } else {
            tvendor = amrex::second()-t0;
        }
    }
#elif defined(AMREX_USE_HIP)
    for (int i = 0; i < 2; ++i) {
        double t0 = amrex::second();
        void * d_temp_storage = nullptr;
        size_t temporary_storage_bytes = 0;
        rocprim::reduce(d_temp_storage, temporary_storage_bytes,
                        p, dsum, npts, rocprim::plus<value_type>());
        d_temp_storage = The_Arena()->alloc(temporary_storage_bytes);
        rocprim::reduce(d_temp_storage, temporary_storage_bytes,
                        p, dsum, npts, rocprim::plus<value_type>());
        Gpu::dtoh_memcpy(hsum, dsum, sizeof(value_type));
        The_Arena()->free(d_temp_storage);
        if (i == 0) {
            amrex::Print() << "    hip sum             = " << *hsum << std::endl;
        } else {
            tvendor = amrex::second()-t0;
        }
    }
#endif

    {
        *hsum = Reduce::Sum(npts, p);
        amrex::Print() << "    Reduce::Sum(ptr)    = " << *hsum << std::endl;
    }
    {
        double t0 = amrex::second();
        *hsum = Reduce::Sum(npts, p);
        tvec = amrex::second()-t0;
    }

    {
        *hsum = Reduce::Sum<value_type>(npts,
                                        [=] AMREX_GPU_DEVICE (int i) -> value_type
                                        { return p[i]; } );
        amrex::Print() << "    Reduce::Sum(lambda) = " << *hsum << std::endl;
    }
    {
        double t0 = amrex::second();
        *hsum = Reduce::Sum<value_type>(npts,
                                        [=] AMREX_GPU_DEVICE (int i) -> value_type
                                        { return p[i]; } );
        t1d = amrex::second()-t0;
    }

    amrex::Print() << "    Kernel run time is " << std::scientific << t << " " << tvendor
                   << " " << tvec << " " << t1d << ".\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int ncell = 256;
        int max_grid_size;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};

        amrex::Print() << "MultiFab::sum:\n";
        test_reduce_sum<MultiFab>(ba,dm,domain);

        amrex::Print() << "iMultiFab::sum\n";
        test_reduce_sum<iMultiFab>(ba,dm,domain);
    }
    amrex::Finalize();
}
