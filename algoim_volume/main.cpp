#include <AMReX.H>
#include <AMReX_algoim.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int n_cell = 128;
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
        }

        Real volume_exact;
        {
            ParmParse pp("eb2");
            Real r;
            pp.get("sphere_radius", r);
            volume_exact = Real(4./3.) * amrex::Math::pi<Real>() * (r*r*r);
        }
        amrex::Print() << "\nExact volume: " << volume_exact << "\n";

        Geometry geom(Box(IntVect(0), IntVect(n_cell-1)));
        EB2::Build(geom, 0, 0, 0);

        BoxArray ba(geom.Domain());
        ba.maxSize(32);
        DistributionMapping dm(ba);

        auto factory = makeEBFabFactory(geom, ba, dm, {0,0,1}, EBSupport::volume);
        auto const& volfrac_eb = factory->getVolFrac();
        auto volume_eb = volfrac_eb.sum(0) * AMREX_D_TERM(geom.CellSize(0),
                                                         *geom.CellSize(1),
                                                         *geom.CellSize(2));
        amrex::Print() << "EV volume: " << volume_eb
                       << " abs error: " << volume_eb - volume_exact 
                       << " rel error: " << std::abs(volume_eb-volume_exact)/volume_exact
                       << "\n";

        MultiFab volfrac_algoim(ba, dm, 1, 0);
        {
            ParmParse pp("eb2");

            RealArray center;
            pp.get("sphere_center", center);

            Real radius;
            pp.get("sphere_radius", radius);

            bool has_fluid_inside;
            pp.get("sphere_has_fluid_inside", has_fluid_inside);

            EB2::SphereIF sf(radius, center, has_fluid_inside);

            algoim::compute_volume_fraction(volfrac_algoim, IntVect(0), geom, sf);
        }
        auto volume_algoim = volfrac_algoim.sum(0) * AMREX_D_TERM(geom.CellSize(0),
                                                                 *geom.CellSize(1),
                                                                 *geom.CellSize(2));
        amrex::Print() << "EV volume: " << volume_algoim
                       << " abs error: " << volume_algoim - volume_exact 
                       << " rel error: " << std::abs(volume_algoim-volume_exact)/volume_exact
                       << "\n";

        amrex::Print() << "\n";
    }
    amrex::Finalize();
}
