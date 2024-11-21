#include <AMReX_FFT.H> // Put this at the top for testing

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#ifdef USE_HEFFTE
#include <heffte.h>
#endif

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");

        int n_cell_x = 256;
        int n_cell_y = 256;
        int n_cell_z = 256;

        Real prob_lo_x = 0.;
        Real prob_lo_y = 0.;
        Real prob_lo_z = 0.;
        Real prob_hi_x = 1.;
        Real prob_hi_y = 1.;
        Real prob_hi_z = 1.;

        {
            ParmParse pp;
            pp.query("n_cell_x", n_cell_x);
            pp.query("n_cell_y", n_cell_y);
            pp.query("n_cell_z", n_cell_z);
        }

        Box domain(IntVect(0),IntVect(AMREX_D_DECL(n_cell_x-1,n_cell_y-1,n_cell_z-1)));
        BoxArray ba = amrex::decompose(domain, ParallelDescriptor::NProcs());
        DistributionMapping dm(ba);

        Geometry geom;
        {
            geom.define(domain,
                        RealBox({AMREX_D_DECL(prob_lo_x,prob_lo_y,prob_lo_z)},
                                {AMREX_D_DECL(prob_hi_x,prob_hi_y,prob_hi_z)}),
                        CoordSys::cartesian, {AMREX_D_DECL(1,1,1)});
        }
        auto const& dx = geom.CellSizeArray();

        MultiFab rhs(ba,dm,1,0);
        MultiFab soln(ba,dm,1,0);
        auto const& rhsma = rhs.arrays();
        ParallelFor(rhs, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            Real x = (i+0.5_rt) * dx[0];
            Real y = (AMREX_SPACEDIM>=2) ? (j+0.5_rt) * dx[1] : 0._rt;
            Real z = (AMREX_SPACEDIM==3) ? (k+0.5_rt) * dx[2] : 0._rt;
            rhsma[b](i,j,k) = std::exp(-10._rt*((x-0.5_rt)*(x-0.5_rt)*1.05_rt +
                                                (y-0.5_rt)*(y-0.5_rt)*0.90_rt +
                                                (z-0.5_rt)*(z-0.5_rt)));
        });

        // Shift rhs so that its sum is zero.
        auto rhosum = rhs.sum(0);
        rhs.plus(-rhosum/geom.Domain().d_numPts(), 0, 1);

        Real facx = 2._rt*Math::pi<Real>()/std::abs(prob_hi_x-prob_lo_x);
        Real facy = 2._rt*Math::pi<Real>()/std::abs(prob_hi_y-prob_lo_y);
        Real facz = 2._rt*Math::pi<Real>()/std::abs(prob_hi_z-prob_lo_z);
        Real scale = 1._rt/(Real(n_cell_z)*Real(n_cell_y)*Real(n_cell_z));

        auto post_forward = [=] AMREX_GPU_DEVICE (int i, int j, int k,
                                                  GpuComplex<Real>& spectral_data)
        {
            amrex::ignore_unused(j,k);
            // the values in the upper-half of the spectral array in y and z
            // are here interpreted as negative wavenumbers
            AMREX_D_TERM(Real a = facx*i;,
                         Real b = (j < n_cell_y/2) ? facy*j : facy*(n_cell_y-j);,
                         Real c = (k < n_cell_z/2) ? facz*k : facz*(n_cell_z-k));
            Real k2 = AMREX_D_TERM(2._rt*(std::cos(a*dx[0])-1._rt)/(dx[0]*dx[0]),
                                  +2._rt*(std::cos(b*dx[1])-1._rt)/(dx[1]*dx[1]),
                                  +2._rt*(std::cos(c*dx[2])-1._rt)/(dx[2]*dx[2]));
            if (k2 != 0._rt) {
                spectral_data /= k2;
            } else {
                // interpretation here is that the average value of the
                // solution is zero
                spectral_data *= 0._rt;
            }
            spectral_data *= scale;
        };

#ifdef USE_HEFFTE
        Box local_box;
        int local_boxid;
        {
            for (int i = 0; i < ba.size(); ++i) {
                Box b = ba[i];
                // each MPI rank has its own local_box Box and local_boxid ID
                if (ParallelDescriptor::MyProc() == dm[i]) {
                    local_box = b;
                    local_boxid = i;
                }
            }
        }

        Box c_local_box = amrex::coarsen(local_box, IntVect(AMREX_D_DECL(2,1,1)));

        // if the coarsened box's high-x index is even, we shrink the size in 1 in x
        // this avoids overlap between coarsened boxes
        if (c_local_box.bigEnd(0) * 2 == local_box.bigEnd(0)) {
            c_local_box.setBig(0,c_local_box.bigEnd(0)-1);
        }
        // for any boxes that touch the hi-x domain we
        // increase the size of boxes by 1 in x
        // this makes the overall fft dataset have size (Nx/2+1 x Ny x Nz)
        if (local_box.bigEnd(0) == geom.Domain().bigEnd(0)) {
            c_local_box.growHi(0,1);
        }

        // each MPI rank gets storage for its piece of the fft
        BaseFab<GpuComplex<Real> > spectral_field(c_local_box, 1, The_Device_Arena());

        auto t0 = amrex::second();

#if defined(AMREX_USE_CUDA)
        heffte::fft3d_r2c<heffte::backend::cufft> fft
#else
        heffte::fft3d_r2c<heffte::backend::fftw> fft
#endif
            ({{local_box.smallEnd(0),local_box.smallEnd(1),local_box.smallEnd(2)},
              {local_box.bigEnd(0)  ,local_box.bigEnd(1)  ,local_box.bigEnd(2)}},
             {{c_local_box.smallEnd(0),c_local_box.smallEnd(1),c_local_box.smallEnd(2)},
              {c_local_box.bigEnd(0)  ,c_local_box.bigEnd(1)  ,c_local_box.bigEnd(2)}},
                0, ParallelDescriptor::Communicator());

        using heffte_complex = typename heffte::fft_output<Real>::type;
        heffte_complex* spectral_data = (heffte_complex*) spectral_field.dataPtr();

        double theffte = 1.e50;
        for (int itry = 0; itry < 4; ++itry)
        {
            auto ta = amrex::second();
            fft.forward(rhs[local_boxid].dataPtr(), spectral_data);
            Array4< GpuComplex<Real> > spectral = spectral_field.array();
            ParallelFor(c_local_box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                post_forward(i,j,k,spectral(i,j,k));
            });
            Gpu::streamSynchronize();
            fft.backward(spectral_data, soln[local_boxid].dataPtr());
            auto tb = amrex::second();
            theffte = amrex::min(theffte, tb-ta);
        }

        auto t1 = amrex::second();
        amrex::Print() << "HEFFTE time: " << t1-t0 << " " << theffte << "\n";

#else
        auto t0 = amrex::second();
        FFT::R2C fft(geom.Domain());
        double tamrex = 1.e50;
        for (int itry = 0; itry < 4; ++itry) {
            auto ta = amrex::second();
            fft.forwardThenBackward(rhs, soln, post_forward);
            auto tb = amrex::second();
            tamrex = amrex::min(tamrex, tb-ta);
        }
        auto t1 = amrex::second();
        amrex::Print() << "AMReX time: " << t1-t0 << " " << tamrex << "\n";
#endif

        {
            MultiFab phi(soln.boxArray(), soln.DistributionMap(), 1, 1);
            MultiFab res(soln.boxArray(), soln.DistributionMap(), 1, 0);
            MultiFab::Copy(phi, soln, 0, 0, 1, 0);
            phi.FillBoundary(geom.periodicity());
            auto const& res_ma = res.arrays();
            auto const& phi_ma = phi.const_arrays();
            auto const& rhs_ma = rhs.const_arrays();
            ParallelFor(res, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
            {
                auto const& phia = phi_ma[b];
                auto lap = (phia(i-1,j,k)-2.*phia(i,j,k)+phia(i+1,j,k)) / (dx[0]*dx[0])
                    +      (phia(i,j-1,k)-2.*phia(i,j,k)+phia(i,j+1,k)) / (dx[1]*dx[1])
                    +      (phia(i,j,k-1)-2.*phia(i,j,k)+phia(i,j,k+1)) / (dx[2]*dx[2]);
                res_ma[b](i,j,k) = rhs_ma[b](i,j,k) - lap;
            });
            amrex::Print() << " rhs.min & max: " <<  rhs.min(0) << " " <<  rhs.max(0) << "\n"
                           << " res.min & max: " <<  res.min(0) << " " <<  res.max(0) << "\n";
//            VisMF::Write(soln, "soln");
//            VisMF::Write(rhs, "rhs");
//            VisMF::Write(res, "res");
        }
    }
    amrex::Finalize();
}
