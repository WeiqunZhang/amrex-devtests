
#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB_utils.H>
#include <AMReX_ParmParse.H>

#include "MyEB.H"

using namespace amrex;

void initializeEB (Geometry const& geom)
{
    ParmParse pp("eb2");
    std::string geom_type;
    pp.get("geom_type", geom_type);

    if (geom_type == "rotated_box")
    {
        EB2::BoxIF box({AMREX_D_DECL(0.45,0.45,0.45)},
                       {AMREX_D_DECL(0.55,0.55,0.55)}, false);
        auto gshop = EB2::makeShop(EB2::translate(
                                       EB2::rotate(
                                           EB2::translate(box, {AMREX_D_DECL(-0.5,-0.5,-0.5)}),
                                           std::atan(1.0)*0.3, 2),
                                       {AMREX_D_DECL(0.5,0.5,0.5)}));
        EB2::Build(gshop, geom, 0, 0);
    }
    else if (geom_type == "flower")
    {
        FlowerIF flower(0.3, 0.15, 6, {AMREX_D_DECL(0.5,0.5,0.5)}, false);
#if (AMREX_SPACEDIM == 2)
        auto gshop = EB2::makeShop(flower);
#else
        EB2::PlaneIF planelo({0.,0.,0.1},{0.,0., -1.});
        EB2::PlaneIF planehi({0.,0.,0.9},{0.,0.,  1.});
        auto gshop = EB2::makeShop(EB2::makeUnion(flower,planelo,planehi));
#endif
        EB2::Build(gshop, geom, 0, 0);
    }
    else
    {
        EB2::Build(geom, 0, 0);
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        int n_cell = 128;
        int max_grid_size = 64;
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect{AMREX_D_DECL(0,0,0)},
                   IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);

        DistributionMapping dm{ba};

        Geometry geom;
        {
            RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
            std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(0,0,0)};
            Geometry::Setup(&rb, 0, isperiodic.data());
            geom.define(domain);
        }

        initializeEB(geom);

        const auto factory = makeEBFabFactory(geom, ba, dm, {4,4,4,4}, EBSupport::full);

        MultiFab mf(amrex::convert(ba,IntVect(1)), dm, 1, 1, MFInfo{}, *factory);

        FillSignedDistance(mf);

        VisMF::Write(mf, "mf");
    }

    amrex::Finalize();
}
