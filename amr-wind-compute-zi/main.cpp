#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>

int compute_zi (amrex::MultiFab const& mf, int dir, amrex::Real dnval,
                amrex::Box const& domain)
{
    AMREX_ALWAYS_ASSERT(domain.smallEnd() == 0); // We could relax this if necessary.
    amrex::Array<bool,AMREX_SPACEDIM> decomp{AMREX_D_DECL(true,true,true)};
    decomp[dir] = false; // no domain decompose in the dir direction.
    auto new_ba = amrex::decompose(domain, amrex::ParallelDescriptor::NProcs(), decomp);

    amrex::Vector<int> pmap(new_ba.size());
    std::iota(pmap.begin(), pmap.end(), 0);
    amrex::DistributionMapping new_dm(std::move(pmap));

    amrex::MultiFab new_mf(new_ba, new_dm, 1, 0);
    new_mf.ParallelCopy(mf, dir, 0, 1);

    amrex::Real zi_sum = 0;
    int myproc = amrex::ParallelDescriptor::MyProc();
    if (myproc < new_mf.size()) {
        auto const& a = new_mf.const_array(myproc);
        amrex::Box box2d = amrex::makeSlab(amrex::Box(a), dir, 0);
        AMREX_ALWAYS_ASSERT(dir == 2); // xxxxx TODO: we can support other directions later
        // xxxxx TODO: sycl can be supported in the future.
        // xxxxx TODO: we can support CPU later.
        int nblocks = box2d.numPts();
        constexpr int nthreads = 128;
        int lenx = box2d.length(0);
        int lenz = domain.length(2);
        int lox = box2d.smallEnd(0);
        int loy = box2d.smallEnd(1);
        amrex::Gpu::DeviceVector<int> tmp(nblocks);
        auto* ptmp = tmp.data();
        amrex::launch<nthreads>(nblocks, amrex::Gpu::gpuStream(),
                                [=] AMREX_GPU_DEVICE()
        {
            int j = int(blockIdx.x) /   lenx;
            int i = int(blockIdx.x) - j*lenx;
            i += lox;
            j += loy;
            amrex::KeyValuePair<amrex::Real,int> r{std::numeric_limits<amrex::Real>::lowest(),0};
            for (int k = threadIdx.x; k < lenz; k += nthreads) {
                if (a(i,j,k) > r.first()) { r.second() = k; }
            }
            r = amrex::Gpu::blockReduceMax<nthreads>(r);
            if (threadIdx.x == 0) {
                ptmp[blockIdx.x] = r.second();
            }
        });

        zi_sum = amrex::Reduce::Sum<amrex::Real>
            (nblocks, [=] AMREX_GPU_DEVICE (int iblock)
                {
                    return (ptmp[iblock] + amrex::Real(0.5)) * dnval;
                });
    }

    amrex::ParallelReduce::Sum(zi_sum, amrex::ParallelDescriptor::IOProcessorNumber(),
                               amrex::ParallelDescriptor::Communicator());

    amrex::Long npts = 1;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (idim != dir) { npts *= domain.length(idim); }
    }
    return zi_sum / static_cast<amrex::Real>(npts);
}

using namespace amrex;

void main_main ()
{
    int n_cell = 128;
    Box domain(IntVect(0), IntVect(n_cell-1));
    RealBox rb({0.,0.,0.}, {1.,1.,1.});
    Geometry geom(domain, rb, 0, {0,0,0});
    BoxArray ba(domain);
    ba.maxSize(32);
    DistributionMapping dm{ba};

    MultiFab mf(ba,dm,3,0);
    amrex::FillRandom(mf, 0, mf.nComp());

    int dir = 2;
    Real dnval = geom.CellSize(dir);

    int m_zi = compute_zi(mf, dir, dnval, domain);
    // only the I/O proc. has a valid value for m_zi!
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    main_main();
    amrex::Finalize();
}
