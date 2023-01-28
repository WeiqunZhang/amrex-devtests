#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <limits>

using namespace amrex;

void stream_init (FArrayBox& av, FArrayBox& bv, FArrayBox& cv)
{
    Box const& box = av.box();
    auto const& a = av.array();
    auto const& b = bv.array();
    auto const& c = cv.array();
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        a(i,j,k) = 1.0;
        b(i,j,k) = 2.0;
        c(i,j,k) = 0.0;
    });
}

void stream_copy (FArrayBox const& av, FArrayBox& cv)
{
    Box const& box = av.box();
    auto const& a = av.array();
    auto const& c = cv.array();
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        c(i,j,k) = a(i,j,k);
    });
}

void stream_scale (FArrayBox& bv, FArrayBox const& cv, Real scalar)
{
    Box const& box = bv.box();
    auto const& b = bv.array();
    auto const& c = cv.array();
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        b(i,j,k) = scalar*c(i,j,k);
    });
}

void stream_add (FArrayBox const& av, FArrayBox const& bv, FArrayBox& cv)
{
    Box const& box = av.box();
    auto const& a = av.array();
    auto const& b = bv.array();
    auto const& c = cv.array();
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        c(i,j,k) = a(i,j,k) + b(i,j,k);
    });
}

void stream_triad (FArrayBox& av, FArrayBox const& bv, FArrayBox const& cv,
                   Real scalar)
{
    Box const& box = av.box();
    auto const& a = av.array();
    auto const& b = bv.array();
    auto const& c = cv.array();
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
        a(i,j,k) = b(i,j,k) + scalar*c(i,j,k);
    });
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv,true,MPI_COMM_WORLD, [] () {
        ParmParse pp("amrex");
        pp.add("the_arena_is_managed", 0);
    });
    amrex::SetVerbose(0);
    {
        int n_cell = 512;
        ParmParse pp;
        pp.query("n_cell", n_cell);

        Box box(IntVect(0), IntVect(n_cell-1));
        FArrayBox faba(box,1);
        FArrayBox fabb(box,1);
        FArrayBox fabc(box,1);
        Real scalar = 3.0;

        stream_init(faba,fabb,fabc);
        Gpu::synchronize();

        // warm up
        stream_copy(faba,fabc);
        stream_scale(fabb,fabc,scalar);
        stream_add(faba,fabb,fabc);
        stream_triad(faba,fabb,fabc,scalar);
        Gpu::synchronize();

        stream_init(faba,fabb,fabc);
        Gpu::synchronize();

        Array2D<double,0,3,0,NTIMES-1,Order::C> times;
        for (int it = 0; it < NTIMES; ++it) {
            Real t = amrex::second();
            stream_copy(faba, fabc);
            Gpu::synchronize();
            times(0,it) = amrex::second() - t;

            t = amrex::second();
            stream_scale(fabb, fabc, scalar);
            Gpu::synchronize();
            times(1,it) = amrex::second() - t;

            t = amrex::second();
            stream_add(faba, fabb, fabc);
            Gpu::synchronize();
            times(2,it) = amrex::second() - t;

            t = amrex::second();
            stream_triad(faba, fabb, fabc, scalar);
            Gpu::synchronize();
            times(3,it) = amrex::second() - t;
        }

        Array<double,4> avgtime, mintime, maxtime;
        for (int j = 0; j < 4; ++j) {
            avgtime[j] = 0.0;
            mintime[j] = std::numeric_limits<Real>::max();
            maxtime[j] = std::numeric_limits<Real>::lowest();
            for (int it = 0; it < NTIMES; ++it) {
                avgtime[j] += times(j,it);
                mintime[j] = amrex::min(mintime[j], times(j,it));
                maxtime[j] = amrex::max(maxtime[j], times(j,it));
            }
            avgtime[j] /= double(NTIMES);
        }

        double bytes[4] = {
            static_cast<double>(2 * sizeof(Real) * box.numPts()),
            static_cast<double>(2 * sizeof(Real) * box.numPts()),
            static_cast<double>(3 * sizeof(Real) * box.numPts()),
            static_cast<double>(3 * sizeof(Real) * box.numPts())
        };

        char label[4][12] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

        for (int iproc = 0; iproc < ParallelDescriptor::NProcs(); ++iproc) {
            ParallelDescriptor::Barrier();
            if (iproc == ParallelDescriptor::MyProc()) {
                if (ParallelDescriptor::NProcs() > 1) {
                    std::cout << "Proc. " << iproc << std::endl;
                }
                std::printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
                for (int j = 0; j < 4; ++j) {
                    std::printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
                                1.0e-06 * bytes[j]/mintime[j], avgtime[j], mintime[j], maxtime[j]);
                }
                std::cout << std::endl;
            }
        }
    }
    amrex::Finalize();
}
