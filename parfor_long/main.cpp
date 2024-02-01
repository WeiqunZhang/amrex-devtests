#include <AMReX.H>
#include <AMReX_FArrayBox.H>

using namespace amrex;

void fold (FArrayBox& fab)
{
    using N = int;
    auto const& a = fab.array();
    Box const& box = fab.box();
    const auto ncells = N(box.numPts());
    auto const& ec = Gpu::makeExecutionConfig<256>(ncells);
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto lenxy = N(len.x)*N(len.y);
    const auto lenx = N(len.x);
    AMREX_LAUNCH_KERNEL(256, ec.numBlocks, ec.numThreads, 0, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE () noexcept {
        for (N icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride)
        {
            N k =  icell /   lenxy;
            N j = (icell - k*lenxy) /   lenx;
            N i = (icell - k*lenxy) - j*lenx;
            a(int(i)+lo.x, int(j)+lo.y, int(k)+lo.z) = 3.;
        }
    });
}

void fnew (FArrayBox& fab)
{
    using N = Long;
    auto const& a = fab.array();
    Box const& box = fab.box();
    const auto ncells = N(box.numPts());
    auto const& ec = Gpu::makeExecutionConfig<256>(ncells);
    const auto lo  = amrex::lbound(box);
    const auto len = amrex::length(box);
    const auto lenxy = N(len.x)*N(len.y);
    const auto lenx = N(len.x);
    AMREX_LAUNCH_KERNEL(256, ec.numBlocks, ec.numThreads, 0, Gpu::gpuStream(),
    [=] AMREX_GPU_DEVICE () noexcept {
        for (N icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
             icell < ncells; icell += stride)
        {
            N k =  icell /   lenxy;
            N j = (icell - k*lenxy) /   lenx;
            N i = (icell - k*lenxy) - j*lenx;
            a(int(i)+lo.x, int(j)+lo.y, int(k)+lo.z) = 3.;
        }
    });
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box box(IntVect(0),IntVect(127));
        FArrayBox fab(box);
        fold(fab);
        fnew(fab);
    }
    amrex::Finalize();
}
