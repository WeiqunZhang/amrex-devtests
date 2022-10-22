// 1: very fast (e.g., 10 seconds)
// 2: slow (e.g., 1 minute)
// 3: very slow (e.g. 1 hour)
#define COMPILE_OPTION 3

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace CoarsenIO {

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    Real Interp ( Array4<Real const> const& arr_src,
                  GpuArray<int,3> const& sf,
                  GpuArray<int,3> const& sc,
                  GpuArray<int,3> const& cr,
                  const int i,
                  const int j,
                  const int k,
                  const int comp )
    {
        // Indices of destination array (coarse)
        const int ic[3] = { i, j, k };

        // Number of points and starting indices of source array (fine)
        int np[3], idx_min[3];

        // Compute number of points
        for ( int l = 0; l < 3; ++l ) {
            if ( cr[l] == 1 ) np[l] = 1+amrex::Math::abs(sf[l]-sc[l]); // no coarsening
            else              np[l] = 2-sf[l];
        }

        // Compute starting indices of source array (fine)
        for ( int l = 0; l < 3; ++l ) {
            if ( cr[l] == 1 ) idx_min[l] = ic[l]-sc[l]*(1-sf[l]); // no coarsening
            else              idx_min[l] = ic[l]*cr[l]+static_cast<int>(cr[l]/2)*(1-sc[l])-(1-sf[l]);
        }

        // Auxiliary integer variables
        const int numx = np[0];
        const int numy = np[1];
        const int numz = np[2];
        const int imin = idx_min[0];
        const int jmin = idx_min[1];
        const int kmin = idx_min[2];
        int  ii, jj, kk;
        Real wx, wy, wz;

        // Interpolate over points computed above
        Real c = 0.0_rt;
        for         (int kref = 0; kref < numz; ++kref) {
            for     (int jref = 0; jref < numy; ++jref) {
                for (int iref = 0; iref < numx; ++iref) {
                    ii = imin+iref;
                    jj = jmin+jref;
                    kk = kmin+kref;
                    wx = 1.0_rt/static_cast<Real>(numx);
                    wy = 1.0_rt/static_cast<Real>(numy);
                    wz = 1.0_rt/static_cast<Real>(numz);
                    c += wx*wy*wz*arr_src(ii,jj,kk,comp);
                }
            }
        }
        return c;
    }
}

void test (MultiFab const& Ex, MultiFab const& Ey, MultiFab const& Ez,
           MultiFab const& Bx, MultiFab const& By, MultiFab const& Bz)
{
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    amrex::ReduceData<Real, Real, Real> reduce_data(reduce_ops);

    const amrex::GpuArray<int,3> cc{0,0,0};
    const amrex::GpuArray<int,3> cr{1,1,1};
    constexpr int comp = 0;

    amrex::GpuArray<int,3> Ex_stag{0,0,0};
    amrex::GpuArray<int,3> Ey_stag{0,0,0};
    amrex::GpuArray<int,3> Ez_stag{0,0,0};
    amrex::GpuArray<int,3> Bx_stag{0,0,0};
    amrex::GpuArray<int,3> By_stag{0,0,0};
    amrex::GpuArray<int,3> Bz_stag{0,0,0};
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        Ex_stag[i] = Ex.ixType()[i];
        Ey_stag[i] = Ey.ixType()[i];
        Ez_stag[i] = Ez.ixType()[i];
        Bx_stag[i] = Bx.ixType()[i];
        By_stag[i] = By.ixType()[i];
        Bz_stag[i] = Bz.ixType()[i];
    }
    for (amrex::MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box & box = enclosedCells(mfi.nodaltilebox());
        const amrex::Array4<const amrex::Real> & Ex_arr = Ex[mfi].array();
        const amrex::Array4<const amrex::Real> & Ey_arr = Ey[mfi].array();
        const amrex::Array4<const amrex::Real> & Ez_arr = Ez[mfi].array();
        const amrex::Array4<const amrex::Real> & Bx_arr = Bx[mfi].array();
        const amrex::Array4<const amrex::Real> & By_arr = By[mfi].array();
        const amrex::Array4<const amrex::Real> & Bz_arr = Bz[mfi].array();

        reduce_ops.eval(box, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        -> amrex::GpuTuple<Real, Real, Real>
        {
            const amrex::Real Ex_cc = CoarsenIO::Interp(Ex_arr, Ex_stag, cc, cr,
                                                        i, j, k, comp);
            const amrex::Real Ey_cc = CoarsenIO::Interp(Ey_arr, Ey_stag, cc, cr,
                                                        i, j, k, comp);
            const amrex::Real Ez_cc = CoarsenIO::Interp(Ez_arr, Ez_stag, cc, cr,
                                                        i, j, k, comp);
            const amrex::Real Bx_cc = CoarsenIO::Interp(Bx_arr, Bx_stag, cc, cr,
                                                        i, j, k, comp);

#if defined(COMPILE_OPTION) && (COMPILE_OPTION >= 2)
            const amrex::Real By_cc = CoarsenIO::Interp(By_arr, By_stag, cc, cr,
                                                        i, j, k, comp);
#else
            const amrex::Real By_cc = By_arr(i,j,k);
#endif

#if defined(COMPILE_OPTION) && (COMPILE_OPTION >= 3)
            const amrex::Real Bz_cc = CoarsenIO::Interp(Bz_arr, Bz_stag, cc, cr,
                                                        i, j, k, comp);
#else
            const amrex::Real Bz_cc = Bz_arr(i,j,k);
#endif

            return {Ey_cc * Bz_cc - Ez_cc * By_cc,
                    Ez_cc * Bx_cc - Ex_cc * Bz_cc,
                    Ex_cc * By_cc - Ey_cc * Bx_cc};
        });
    }

    auto r = reduce_data.value();
    amrex::Real ExB_x = amrex::get<0>(r);
    amrex::Real ExB_y = amrex::get<1>(r);
    amrex::Real ExB_z = amrex::get<2>(r);
    amrex::Print() << "Results: " << ExB_x << " " << ExB_y << " " << ExB_z << "\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ncell = 256;
        int max_grid_size;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};

        MultiFab Ex(ba,dm,1,0);
        MultiFab Ey(ba,dm,1,0);
        MultiFab Ez(ba,dm,1,0);
        MultiFab Bx(ba,dm,1,0);
        MultiFab By(ba,dm,1,0);
        MultiFab Bz(ba,dm,1,0);
        Ex.setVal(1.);
        Ey.setVal(1.);
        Ez.setVal(1.);
        Bx.setVal(1.);
        By.setVal(1.);
        Bz.setVal(1.);
        test(Ex,Ey,Ez,Bx,By,Bz);
    }
    amrex::Finalize();
}
