#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Vector.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ntests = 100;
        int maxboxes = 16;
        int maxblocks = 64;
        ParmParse pp;
        pp.query("ntests", ntests);
        pp.query("maxboxes", maxboxes);
        pp.query("maxblocks", maxblocks);

        for (int itest = 0; itest < ntests; ++itest) {
            int nboxes = amrex::Random_int(16) + 1;
            Vector<int> nblocks(nboxes+1);
            nblocks[0] = 0;
            for (int i = 0; i < nboxes; ++i) {
                nblocks[i+1] = nblocks[i] + amrex::Random_int(maxblocks);
            }
            int ntotblocks = nblocks[nboxes];
            for (int bid = 0; bid < ntotblocks; ++bid) {
                int ibox = amrex::bisect(nblocks.data(), 0, nboxes, bid);
                AMREX_ALWAYS_ASSERT(nblocks[ibox] <= bid && bid < nblocks[ibox+1]);
            }
        }
    }
    amrex::Finalize();
}
