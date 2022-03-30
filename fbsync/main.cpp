#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

static
BoxArray
readBoxList (const std::string& file, Box& domain)
{
    BoxList retval;

    std::ifstream boxspec;

    boxspec.open(file.c_str(), std::ios::in);

    if( !boxspec )
    {
        std::string msg = "readBoxList: unable to open ";
        msg += file;
        amrex::Error(msg.c_str());
    }
    boxspec >> domain;
    
    int numbox = 0;
    boxspec >> numbox;

    for ( int i=0; i<numbox; i++ )
    {
        Box tmpbox;
        boxspec >> tmpbox;
        if( !domain.contains(tmpbox) )
	{
            std::cerr << "readBoxList: bogus box " << tmpbox << '\n';
            exit(1);
        }
        retval.push_back(tmpbox);
    }

    return BoxArray(retval);
}

void test(int ncell, int max_grid_size, int nghost,
          Array<int,AMREX_SPACEDIM> const& is_periodic)
{
    Box domain(IntVect(0), IntVect(ncell-1));
    BoxArray grids(domain);
    grids.maxSize(max_grid_size);
    grids.convert(IntVect(1));
    DistributionMapping dmap(grids);
    Geometry geom(domain, RealBox(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)),
                  CoordSys::cartesian, is_periodic);

    MultiFab mf(grids, dmap, 1, 2);
    MultiFab mf2(grids, dmap, 1, 2);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        mf[mfi].setVal(static_cast<Real>(mfi.index()));
        mf2[mfi].setVal(static_cast<Real>(mfi.index()));
    }
    
    mf.FillBoundaryAndSync(0, 1, IntVect(nghost), geom.periodicity());

    {
        auto msk = mf2.OwnerMask(geom.periodicity());
        amrex::OverrideSync(mf2, *msk, geom.periodicity());
        mf2.FillBoundary(IntVect(nghost), geom.periodicity());
    }

    MultiFab::Subtract(mf2, mf, 0, 0, 1, nghost);
    Real error = mf2.norminf(0, 1, IntVect(nghost));
    IntVect p{AMREX_D_DECL(is_periodic[0], is_periodic[1], is_periodic[2])};
    if (error == Real(0.0)) {
        amrex::Print() << "FillBoundaryAndSync PASSED test " << ncell << " "
                       << max_grid_size << " " << nghost << " " << p << "\n";
    } else {
        amrex::Print() << "FillBoundaryAndSync FAILED test " << ncell << " "
                       << max_grid_size << " " << nghost << " " << p << "\n";
    }

//    amrex::writeFabs(mf, "mf");
}

#if (AMREX_SPACEDIM == 3)
void test2 (std::string const& file)
{
    Box domain;
    BoxArray grids = readBoxList(file, domain);
    grids.convert(IntVect(1));

    DistributionMapping dmap(grids);
    Geometry geom(domain, RealBox(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)),
                  CoordSys::cartesian, {1,1,1});

    MultiFab mf(grids, dmap, 1, 1);
    MultiFab mf2(grids, dmap, 1, 1);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        mf[mfi].setVal(static_cast<Real>(mfi.index()));
        mf2[mfi].setVal(static_cast<Real>(mfi.index()));
    }
    
    mf.FillBoundaryAndSync(0, 1, IntVect(1), geom.periodicity());

    {
        auto msk = mf2.OwnerMask(geom.periodicity());
        amrex::OverrideSync(mf2, *msk, geom.periodicity());
        mf2.FillBoundary(IntVect(1), geom.periodicity());
    }

    MultiFab::Subtract(mf2, mf, 0, 0, 1, 1);
    Real error = mf2.norminf(0, 1, IntVect(1));
    if (error == Real(0.0)) {
        amrex::Print() << file << " test PASSED.\n";
    } else {
        amrex::Print() << file << " test FAILED.\n";
    }

    amrex::VisMF::Write(mf, "mf");
}
#endif

void main_main ()
{
    int ncell = 256;
    ParmParse pp;
    pp.query("ncell", ncell);

    for (int max_grid_size = 256; max_grid_size >= 32; max_grid_size /= 2) {
        for (int nghost = 0; nghost <= 2; ++nghost) {
            Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                test(ncell, max_grid_size, nghost, is_periodic);
                is_periodic[idim] = 1;
            }
        }
    }

#if (AMREX_SPACEDIM == 3)
    test2("grids.213");
#endif
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    main_main();
    amrex::Finalize();
}
