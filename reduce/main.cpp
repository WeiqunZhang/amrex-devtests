#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

#include <cub/cub.cuh>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Real t, tcub, tvec, t1d;
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
        MultiFab mf(ba,dm,1,0);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            Array4<Real> const& a = mf.array(mfi);
            amrex::ParallelForRNG(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine)
            {
                a(i,j,k) = amrex::Random(engine) + 0.5_rt;
            });
        }
        {
            BL_PROFILE("reduce-warmup");
            mf.sum();
        }
        {
            BL_PROFILE("reduce-mf");
            amrex::Real t0 = amrex::second();
            mf.sum();
            t = amrex::second()-t0;
        }
        Real* p = mf[0].dataPtr();
        Real* hsum = (Real*)The_Pinned_Arena()->alloc(sizeof(Real));
        Real* dsum = (Real*)The_Arena()->alloc(sizeof(Real));
        Long num_items = domain.numPts();
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, dsum, num_items);
            // Allocate temporary storage
            d_temp_storage = (void*)The_Arena()->alloc(temp_storage_bytes);
            // Run sum-reduction
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, dsum, num_items);
            Gpu::dtoh_memcpy(hsum,dsum, sizeof(Real));
            Gpu::synchronize();
        }
        {
            amrex::Real t0 = amrex::second();
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, dsum, num_items);
            // Allocate temporary storage
            d_temp_storage = (void*)The_Arena()->alloc(temp_storage_bytes);
            // Run sum-reduction
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, dsum, num_items);
            Gpu::dtoh_memcpy(hsum,dsum, sizeof(Real));
            Gpu::synchronize();
            tcub = amrex::second()-t0;
        }

        {
            *hsum = Reduce::Sum(num_items, p);
        }
        {
            amrex::Real t0 = amrex::second();
            *hsum = Reduce::Sum(num_items, p);
            tvec = amrex::second()-t0;
        }

        {
            *hsum = Reduce::Sum<Real>(num_items,
                                      [=] AMREX_GPU_DEVICE (int i) -> Real { return p[i]; } );
        }
        {
            amrex::Real t0 = amrex::second();
            *hsum = Reduce::Sum<Real>(num_items,
                                      [=] AMREX_GPU_DEVICE (int i) -> Real { return p[i]; } );
            t1d = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << " " << tcub
              << " " << tvec << " " << t1d << ".\n";
}
