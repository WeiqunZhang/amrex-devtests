#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Geometry.H>

using namespace amrex;

// test n_cells too

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    for (int icell = 0; icell < 10000; ++icell)
    {
        int ncells = amrex::Random_int(102400) + 4;
        Box domain(IntVect(0),IntVect(ncells-1));

        for (int ieps = 0; ieps < 10000; ++ieps)
        {
            std::array<Real,AMREX_SPACEDIM> rblo{AMREX_D_DECL(Real(0.),Real(-1.),Real(-0.3))};
            std::array<Real,AMREX_SPACEDIM> rbhi{AMREX_D_DECL(Real(1.),Real( 0.),Real( 0.5))};
            if (ieps % 100 != 0) {
                auto eps = amrex::Random() * 1.e-4;
                AMREX_D_TERM(rblo[0] += eps;,
                             rblo[1] -= eps;,
                             rblo[2] += eps);
                AMREX_D_TERM(rbhi[0] -= eps;,
                             rbhi[1] += eps;,
                             rbhi[2] -= eps);
            }

            RealBox rb(rblo, rbhi);
            Geometry geom(domain, rb, 0, {AMREX_D_DECL(0,0,0)});

            auto rlo = geom.ProbLoArrayInParticleReal();
            auto rhi = geom.ProbHiArrayInParticleReal();
            auto plo = geom.ProbLoArray();
            auto dxinv = geom.InvCellSizeArray();
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                auto index = [&] (ParticleReal x) -> int
                {
                    return int(std::floor((x - plo[idim])*dxinv[idim]));
                };
                auto epsilon = std::numeric_limits<ParticleReal>::epsilon()
                    * std::max(ParticleReal(geom.CellSize(idim)),std::abs(rlo[idim]))
                    * ParticleReal(2.0);
                auto rlom = rlo[idim] - epsilon;
                epsilon = std::numeric_limits<ParticleReal>::epsilon()
                    * std::max(ParticleReal(geom.CellSize(idim)),std::abs(rhi[idim]))
                    * ParticleReal(2.0);
                auto rhim = rhi[idim] - epsilon;
                bool pass = (index(rlom)      == -1)
                    &&      (index(rlo[idim]) == 0 )
                    &&      (index(rhim)      == ncells-1)
                    &&      (index(rhi[idim]) == ncells);
                if (!pass) {
                    amrex::Print() << "Geometry: " << geom << std::endl;
                    amrex::Print().SetPrecision(17)
                        << " rlo[" << idim << "] = " << rlo[idim]
                        << " rhi[" << idim << "] = " << rhi[idim]
                        << " rlo_minus = " << rlom
                        << " rhi_minus = " << rhim << "\n";
                    amrex::Print() << " ilo = " << index(rlo[idim])
                                   << " ihi = " << index(rhim)
                                   << " ilo-1 = " << index(rlom)
                                   << " ihi+1 = " << index(rhi[idim])
                                   << "\n";
                    //amrex::Abort("Failed");
                }
            }
        }
    }
    amrex::Finalize();
}
