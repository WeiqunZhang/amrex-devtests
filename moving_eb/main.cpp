#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

void build_sphere (Array<Real,AMREX_SPACEDIM> const& center,
                   Real radius, Geometry const& geom)
{
    EB2::SphereIF sf(radius, center, false);
    EB2::GeometryShop<EB2::SphereIF> gshop(sf);
    EB2::Build(gshop, geom, 0, 0, 2);
}

void main_main ()
{
    Box domain(IntVect(0), IntVect(255));
    BoxArray grids(domain);
    grids.maxSize(32);
    DistributionMapping dmap(grids);
    Geometry geom(domain, RealBox(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)),
                  CoordSys::cartesian, {AMREX_D_DECL(0,0,0)});

    Array<Real,AMREX_SPACEDIM> center{AMREX_D_DECL(0. , 0. , 0. )};
    Real const radius = 0.15;
    build_sphere(center, radius, geom);

    EB2::IndexSpace const* eb_new = &(EB2::IndexSpace::top());
    EB2::IndexSpace const* eb_old = nullptr;
    std::unique_ptr<EBFArrayBoxFactory> fact_new = makeEBFabFactory(eb_new, geom,
                                                                    grids, dmap,
                                                                    {1,1,1,1},
                                                                    EBSupport::full);
    std::unique_ptr<EBFArrayBoxFactory> fact_old;
    WriteMLMF(Concatenate("plt",0), {&(fact_new->getVolFrac())}, {geom});
    
    for (int step = 0; step < 50; ++step) {
        EB2::IndexSpace::erase(const_cast<EB2::IndexSpace*>(eb_old));  // erase old EB
        // Build a new EB
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            center[idim] += 0.02;
        }
        build_sphere(center, radius, geom);
        eb_old = eb_new;
        eb_new = &(EB2::IndexSpace::top());
        fact_old = std::move(fact_new);
        fact_new = makeEBFabFactory(eb_new, geom, grids, dmap, {1,1,1,1}, EBSupport::full);

        MultiFab plotmf(grids,dmap,2, 0);
        MultiFab::Copy(plotmf, fact_old->getVolFrac(), 0, 0, 1, 0);
        MultiFab::Copy(plotmf, fact_new->getVolFrac(), 0, 1, 1, 0);

        WriteMLMF(Concatenate("plt",step+1), {&plotmf}, {geom});
    }

    amrex::Print() << "Size of EB IndexSpace at the end is " << EB2::IndexSpace::size() << "\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    main_main();
    amrex::Finalize();
}
