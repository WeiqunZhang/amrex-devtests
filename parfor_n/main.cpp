#include <AMReX.H>
#include <AMReX_FArrayBox.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        const int nfabs = 26;
        Box box(IntVect(0),IntVect(127));
        Box boxg1 = amrex::grow(box,1);

        Vector<FArrayBox> fabs_in(nfabs);
        Vector<FArrayBox> fabs_out(nfabs);
        for (int ifab = 0; ifab < nfabs; ++ifab) {
            fabs_in[ifab].resize(boxg1,1);
            fabs_in[ifab].template setVal<RunOn::Device>(3.14);
            fabs_out[ifab].resize(box,1);
            fabs_out[ifab].template setVal<RunOn::Device>(0.0);
        }

        FArrayBox fabn_in(boxg1, nfabs);
        fabn_in.template setVal<RunOn::Device>(3.14);
        FArrayBox fabn_out(box, nfabs);
        fabn_out.template setVal<RunOn::Device>(0.0);

        Gpu::streamSynchronize();
        auto t0 = amrex::second();

        for (int ifab = 0; ifab < nfabs; ++ifab) {
            auto const& ai = fabs_in[ifab].const_array();
            auto const& ao = fabs_out[ifab].array();
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ao(i,j,k) = 6.*ai(i,j,k) - ai(i-1,j,k) - ai(i+1,j,k)
                    - ai(i,j-1,k) - ai(i,j+1,k) - ai(i,j,k-1) - ai(i,j,k+1);
            });
        }

        Gpu::streamSynchronize();
        auto t1 = amrex::second();

        Gpu::PinnedVector<Array4<Real>> hv(nfabs*2);
        for (int ifab = 0; ifab < nfabs; ++ifab) {
            hv[2*ifab] = fabs_in[ifab].array();
            hv[2*ifab+1] = fabs_out[ifab].array();
        }
        Gpu::Buffer<Array4<Real>> buf(hv.data(), hv.size());
        auto const* pbuf = buf.data();
        amrex::ParallelFor(box, nfabs, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            auto const& ai = pbuf[2*n];
            auto const& ao = pbuf[2*n+1];
            ao(i,j,k) = 6.*ai(i,j,k) - ai(i-1,j,k) - ai(i+1,j,k)
                - ai(i,j-1,k) - ai(i,j+1,k) - ai(i,j,k-1) - ai(i,j,k+1);
        });

        Gpu::streamSynchronize();
        auto t2 = amrex::second();

        auto const& ani = fabn_in.const_array();
        auto const& ano = fabn_out.array();
        amrex::ParallelFor(box, nfabs, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
        {
            ano(i,j,k,n) = 6.*ani(i,j,k,n) - ani(i-1,j,k,n) - ani(i+1,j,k,n)
                - ani(i,j-1,k,n) - ani(i,j+1,k,n) - ani(i,j,k-1,n) - ani(i,j,k+1,n);
        });

        Gpu::streamSynchronize();
        auto t3 = amrex::second();

        amrex::Print() << "  26 kernels                  : " << t1-t0 << "\n"
                       << "  1 kernel on 26 fabs         : " << t2-t1 << "\n"
                       << "  1 kernel on 1 multi-comp fab: " << t3-t2 << "\n";
    }
    amrex::Finalize();
}
