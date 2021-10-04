#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box domain(IntVect(0), IntVect(127));
        BoxArray ba(domain);
        ba.maxSize(IntVect(1024, 32));
        DistributionMapping dm{ba};

        MultiFab mf(ba, dm, 1, 2);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            Box const& gbx = mfi.fabbox();
            Box const& vbx = mfi.validbox();
            int jlo = vbx.smallEnd(1);
            auto const& a = mf.array(mfi);
            amrex::ParallelFor(gbx, [=] (int i, int j, int k)
            {
                a(i,j,k) = j-jlo;
            });
        }

        { // special FillBoundary
            BoxList bl;
            for (int i = 0, N=ba.size(); i < N; ++i) {
                bl.push_back(amrex::grow(ba[i], 0, mf.nGrowVect()[0]));
            }
            BoxArray rba(std::move(bl));
            MultiFab rmf(rba, dm, mf.nComp(), IntVect(0,mf.nGrowVect()[1]), MFInfo().SetAlloc(false));

            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                rmf.setFab(mfi, FArrayBox(mf[mfi], amrex::make_alias, 0, mf.nComp()));
            }

            rmf.FillBoundary();
        }

        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            std::ofstream ofs("fab-"+std::to_string(mfi.index()));
            mf[mfi].writeOn(ofs);
        }
    }
    amrex::Finalize();
}
