#include <AMReX.H>
#include <AMReX_MultiFab.H>

static_assert(AMREX_SPACEDIM == 3);

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box domain(IntVect(0), IntVect(31,31,31));
        BoxList bl;
        IntVect lo, hi;
        for (int k = 0; k < 3; ++k) {
            if (k == 0) {
                lo[2] = 0;
                hi[2] = 7;
            } else if (k == 1) {
                lo[2] = 8;
                hi[2] = 19;
            } else {
                lo[2] = 20;
                hi[2] = 31;
            }
            for (int j = 0; j < 3; ++j) {
                if (j == 0) {
                    lo[1] = 0;
                    hi[1] = 7;
                } else if (j == 1) {
                    lo[1] = 8;
                    hi[1] = 19;
                } else {
                    lo[1] = 20;
                    hi[1] = 31;
                }
                for (int i = 0; i < 3; ++i) {
                    if (i == 0) {
                        lo[0] = 0;
                        hi[0] = 7;
                    } else if (i == 1) {
                        lo[0] = 8;
                        hi[0] = 19;
                    } else {
                        lo[0] = 20;
                        hi[0] = 31;
                    }
                    bl.push_back(Box(lo,hi));
                }
            }
        }
        BoxArray ba(std::move(bl));
        DistributionMapping dm{ba};

        Geometry geom(domain,
                      RealBox(Array<Real,3>{0.0_rt,0.0_rt,0.0_rt},
                              Array<Real,3>{amrex::Math::pi<Real>(),
                                            2.0_rt*amrex::Math::pi<Real>(),
                                            1.0_rt}),
                      0, Array<int,3>{0,1,0});

        auto f = [] (int i, int j, int k) -> GpuArray<Long,4>
        {
            Long ijk = k*Long(1'000'000)+j*Long(1'000)+i;
            return {ijk,
                    ijk + Long(1'000'000'000),
                    ijk + Long(2'000'000'000),
                    ijk + Long(3'000'000'000)};
        };

        FabArray<BaseFab<Long>> mf(ba, dm, 4, 2);
        auto const& ma = mf.arrays();
        amrex::ParallelFor(mf, [=] (int box_no, int i, int j, int k)
        {
            auto v = f(i,j,k);
            auto const& a = ma[box_no];
            a(i,j,k,0) = v[0];
            a(i,j,k,1) = v[1];
            a(i,j,k,2) = v[2];
            a(i,j,k,3) = v[3];
        });

        mf.setDomainBndry(-1, geom);

        SphThetaPhiRMapping index_mapping{domain};

        auto cmd = makeFillBoundaryMetaData(mf, mf.nGrowVect(), geom, index_mapping);
        // cmd can be cached, should be if it's used more than once.
        FillBoundary(mf, cmd, 0, mf.nComp(), index_mapping);

        int nx = domain.length(0);
        int ny = domain.length(1);
        int nz = domain.length(2);
        int nfail = 0;
#if (AMREX_USE_GPU)
        static_assert(false, "This test is not for GPU");
#endif
        amrex::ParallelFor(mf, mf.nGrowVect(), [&] (int box_no, int i, int j, int k)
        {
            if (!((k >= nz) ||
                  (i <   0 && j >= 0  && j < ny && k < 0) ||
                  (i >= nx && j >= 0  && j < ny && k <0) ||
                  (i <   0 && j <  0  && k < 0) ||
                  (i >= nx && j <  0  && k < 0) ||
                  (i <   0 && j >= ny && k < 0) ||
                  (i >= nx && j >= ny && k < 0)))
            {
                auto s = index_mapping(Dim3{i,j,k});
                auto v = f(s.x,s.y,s.z);
                auto const& a = ma[box_no];
                if (a(i,j,k,0) != v[0] ||
                    a(i,j,k,1) != v[1] ||
                    a(i,j,k,2) != v[2] ||
                    a(i,j,k,3) != v[3]) {
                    ++nfail;
                }
            }
        });
        if (nfail > 0) {
            amrex::Print() << "Failed in " << nfail << " cells.\n";
        } else {
            amrex::Print() << "PASS\n";
        }
    }
    amrex::Finalize();
}
