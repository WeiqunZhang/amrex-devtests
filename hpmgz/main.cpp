#include "HpMultiGrid.H"

#include <AMReX.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int type = 2;
        int n_cell = 64;
        ParmParse pp;
        pp.query("type", type);
        pp.query("n_cell", n_cell);

        Box domain(IntVect(0,0,10), IntVect(n_cell-1,n_cell-1,10));
        Geometry geom(domain, RealBox({0.,0.,0.},{1.,1.,1.}), 0, {0,0,0});

        FArrayBox s(domain, 2);
        FArrayBox f(domain, 2);
        FArrayBox a(domain, 1);
        Real ar = 2.;

        {
            auto s_arr = s.array();
            auto f_arr = f.array();
            auto a_arr = a.array();
            auto plo = geom.ProbLoArray();
            auto dx  = geom.CellSizeArray();
            amrex::ParallelFor(domain, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real x = plo[0] + (i+0.5)*dx[0];
                Real y = plo[1] + (j+0.5)*dx[1];
                constexpr Real pi = 3.1415926535897932;
                s_arr(i,j,k,0) = std::sin(2.*pi*x) * std::sin(3.*pi*y);
                s_arr(i,j,k,1) = std::sin(4.*pi*x) * std::sin(2.*pi*y);
                a_arr(i,j,k) = 1.0 + 0.25*std::cos(1.75*pi*x)*std::cos(0.8*pi*y);
                if (type == 1) {
                    f_arr(i,j,k,0) = (-a_arr(i,j,k) - 13.*pi*pi) * s_arr(i,j,k,0);
                    f_arr(i,j,k,1) = (-a_arr(i,j,k) - 20.*pi*pi) * s_arr(i,j,k,1);
                } else {
                    f_arr(i,j,k,0) = -ar*s_arr(i,j,k,0) + a_arr(i,j,k)*s_arr(i,j,k,1)
                        - 13.*pi*pi*s_arr(i,j,k,0);
                    f_arr(i,j,k,1) = -ar*s_arr(i,j,k,1) - a_arr(i,j,k)*s_arr(i,j,k,0)
                        - 20.*pi*pi*s_arr(i,j,k,1);
                }
            });
        }

        FArrayBox s0(domain,2);
        s0.setVal<RunOn::Device>(0.);

        hpmg::MultiGrid mgz(geom);

        if (type == 1) {
            mgz.solve1(s0, f, a, 1.0e-11, 0.0, 100, 2);
        } else {
            mgz.solve2(s0, f, ar, a, 1.0e-11, 0.0, 100, 2);
        }

        {
            std::ofstream ofs("s0");
            s0.writeOn(ofs);
        }
        {
            std::ofstream ofs("s");
            s.writeOn(ofs);
        }

        s0.minus<RunOn::Device>(s);
        amrex::Print() << "Max error: " << s0.maxabs<RunOn::Device>() << std::endl;
    }
    amrex::Finalize();
}
