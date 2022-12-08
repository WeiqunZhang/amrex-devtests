#include <AMReX.H>
#include <AMReX_Gpu.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        const std::size_t size = Gpu::Device::totalGlobalMem() / 2;
        Gpu::DeviceVector<char> dv(size);
        Gpu::DeviceVector<char> ds(1);
        Gpu::PinnedVector<char> hs(1);
        Gpu::synchronize();

        for (int i = 0; i < 100; ++i) {
            Gpu::Device::setStreamIndex(0);
            Vector<char> hv(size, 1);
            ds.assign(1, 2);
            hs.assign(1, 4); // assign is synchronous.

            // expensive copy on stream 0
            Gpu::copyAsync(Gpu::hostToDevice, hv.begin(), hv.end(), dv.begin());

            // cheap copy on stream 0
            Gpu::copyAsync(Gpu::hostToDevice, hs.begin(), hs.end(), ds.begin());

            // a kernel stream 0
            auto pv = dv.data();
            amrex::single_task(Gpu::gpuStream(), [=] AMREX_GPU_DEVICE () {
                pv[size-1] += 5;
            });

            // kernel on stream 1 that depends on
            Gpu::Device::setStreamIndex(1);
            auto p = ds.data();
            amrex::single_task(Gpu::gpuStream(), [=] AMREX_GPU_DEVICE () {
                *p += 10;
            });

            Gpu::streamSynchronizeAll();

            Gpu::copyAsync(Gpu::deviceToHost, ds.begin(), ds.end(), hs.begin());
            Gpu::streamSynchronize();
            if (hs[0] == 4) {
                amrex::Print() << "We have a problem\n";
            } else if (hs[1] != 14) {
                amrex::Print() << "How did this happen?\n";
            }
        }
    }
    amrex::Finalize();
}
