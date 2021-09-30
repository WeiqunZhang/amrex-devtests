#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

double test1 (MultiFab& mf)
{
    double t = amrex::second();
    const int ncomp = mf.nComp();
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        Box const& vbx = mfi.validbox();
        Box const& gbx = mfi.fabbox();
        auto const& fab = mf.array(mfi);
        ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (! vbx.contains(i,j,k)) {
                for (int n = 0; n < ncomp; ++n) {
                    fab(i,j,k,n) = 3.;
                }
            }
        });
    }
    return amrex::second()-t;
}

double test2 (MultiFab& mf)
{
    double t = amrex::second();
    const IntVect& nghost = mf.nGrowVect();
    const int ncomp = mf.nComp();
    auto const& fabs = mf.arrays();
    ParallelFor(mf, nghost,
    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        auto const& fab = fabs[box_no];
        Box vbx(fab);
        vbx.grow(-nghost);
        if (! vbx.contains(i,j,k)) {
            for (int n = 0; n < ncomp; ++n) {
                fab(i,j,k,n) = 3.;
            }
        }
    });
    Gpu::synchronize();
    return amrex::second()-t;
}

double test3 (MultiFab& mf)
{
    double t = amrex::second();
    const IntVect& nghost = mf.nGrowVect();
    const int ncomp = mf.nComp();

    using Tag = Array4BoxTag<Real>;
    Vector<Tag> tags;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        Box const& vbx = mfi.validbox();
        auto const& a = mf.array(mfi);

        Box b = amrex::adjCellLo(vbx, 2, nghost[2]);
        b.grow(IntVect(nghost[0],nghost[1],0));
        tags.emplace_back(Tag{a, b});
        b.shift(2, vbx.length(2)+nghost[2]);
        tags.emplace_back(Tag{a, b});

        b = amrex::adjCellLo(vbx, 1, nghost[1]);
        b.grow(IntVect(nghost[0],0,0));
        tags.emplace_back(Tag{a, b});
        b.shift(1, vbx.length(1)+nghost[1]);
        tags.emplace_back(Tag{a, b});

        b = amrex::adjCellLo(vbx, 0, nghost[0]);
        tags.emplace_back(Tag{a, b});
        b.shift(0, vbx.length(0)+nghost[0]);
        tags.emplace_back(Tag{a, b});
    }

    ParallelFor(tags, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, Tag const& tag) noexcept
    {
        tag.dfab(i,j,k,n) = 3.;
    });

    return amrex::second()-t;
}

double test4 (MultiFab& mf)
{
    double t = amrex::second();
    mf.setBndry(3.0);
    return amrex::second()-t;
}

void test (MultiFab & mf)
{
    double t1, t2, t3, t4;
    for (int itest = 0; itest < 2; ++itest) {
        t1 = test1(mf);
        t2 = test2(mf);
        t3 = test3(mf);
        t4 = test4(mf);
    }
    amrex::Print() << "   Run times are " << std::scientific
                   << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int ncell = 256;
        int max_grid_size;
        int ngrow = 1;
        std::vector<int> box_sizes;
        std::vector<int> nboxes;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.query("max_grid_size", max_grid_size);
            pp.query("ngrow", ngrow);
            pp.queryarr("box_sizes", box_sizes);
            nboxes.resize(box_sizes.size(),1);
            pp.queryarr("nboxes", nboxes);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba;
        if (box_sizes.empty()) {
            ba = BoxArray(domain);
            ba.maxSize(max_grid_size);
        } else {
            BoxList bl;
            for (int i = 0; i < box_sizes.size(); ++i) {
                for (int j = 0; j < nboxes[i]; ++j) {
                    bl.push_back(Box(IntVect(0), IntVect(box_sizes[i]-1)));
                }
            }
            ba = BoxArray(std::move(bl));
        }
        DistributionMapping dm{ba};
        MultiFab mf(ba, dm, 1, ngrow);
        mf.setVal(1.);

        test(mf);
    }
    amrex::Finalize();
}
