#include "HpMultiGrid.H"

#include <AMReX.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuComplex.H>

#include <cmath>

using namespace amrex;
using Complex = amrex::GpuComplex<amrex::Real>;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int type = 2;
        int n_cell = 512;
        ParmParse pp;
        pp.query("type", type);
        pp.query("n_cell", n_cell);

        // Constants
        constexpr Real pi = 3.1415926535897932;
        const Complex I(0,1);
        const Real c = 299'792'458.;
        // Laser parameters
        const Real w0 = 40.e-6;

        Box domain(IntVect(0,0,100), IntVect(n_cell-1,n_cell-1,100));
        Geometry geom(domain, RealBox({
                    -200.e-6,
                    -200.e-6,
                    -  .8e-6},{
                    +200.e-6,
                    +200.e-6,
                    +  .6e-6}), 0, {0,0,0});

        // typical parameters
        auto dx  = geom.CellSizeArray();
        amrex::Real dt = 10.e-15;

        FArrayBox s(domain, 2); // sol
        FArrayBox f(domain, 2); // rhs
        FArrayBox a(domain, 1); // acoef_imag
        Real ar = 1.e2/w0/w0;// -3._rt/(c*dt*dx[2]) + 2._rt/(c*c*dt*dt);
        Real ai = 0.e-1/w0/w0;//0.;// -2._rt*k0 / (c*dt);

        const int imin = domain.smallEnd(0);
        const int imax = domain.bigEnd  (0);
        const int jmin = domain.smallEnd(1);
        const int jmax = domain.bigEnd  (1);
        {
            auto s_arr = s.array(); // sol
            auto f_arr = f.array(); // rhs
            auto a_arr = a.array(); // acoef_imag
            auto plo = geom.ProbLoArray();
            amrex::ParallelFor(domain, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real x = plo[0] + (i+0.5)*dx[0];
                Real y = plo[1] + (j+0.5)*dx[1];
                Real z = plo[2] + (k+0.5_rt)*dx[2];
                
                s_arr(i,j,k,0) = exp(-( x*x + y*y )/w0/w0);
                s_arr(i,j,k,1) = 0.;

                Real lapR = 4.*(-1.+1.*(x*x+y*y)/w0/w0)/w0/w0*s_arr(i,j,k,0);
                Real lapI = 0.;

                a_arr(i,j,k) = ai;
                if (type == 1) {
                    f_arr(i,j,k,0) = lapR;
                    f_arr(i,j,k,1) = lapI;
                } else {
                    // check above if ai and ar are 0 or not.
                    f_arr(i,j,k,0) =
                        + lapR
                        - ar * s_arr(i,j,k,0)
                        + ai * s_arr(i,j,k,1);
                    f_arr(i,j,k,1) =
                        + lapI
                        - ar * s_arr(i,j,k,1)
                        - ai * s_arr(i,j,k,0);
                }
            });
        }

        // Solving lap f - a*f = lap f0 - a*f0
            
        FArrayBox s0(domain,2);
        s0.setVal<RunOn::Device>(0.);

        hpmg::MultiGrid mgz(geom);

        if (type == 1) {
            mgz.solve1(s0, f, a, 1.0e-12, 0.0, 100, 2);
        } else {
            // sol, rhs, acoef_real acoef_imag, tol...
            mgz.solve2(s0, f, ar, a, 1.0e-12, 0.0, 100, 2);
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
        amrex::Print() << "Max solut: " << s .maxabs<RunOn::Device>() << std::endl;
        // amrex::Print() << "Max error: " << s0.sum<RunOn::Device>() << std::endl;
    }
    amrex::Finalize();
}


// amrex::Print()<<diffract_factor<<" -- "<<inv_complex_waist_2<<" -- "<<prefactor<<" -- "<<time_exponent<<" -- "<<stcfactor<<" -- "<<exp_argument<<" -- "<<envelope<<" -- "<<z<<'\n';
