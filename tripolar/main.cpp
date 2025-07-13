#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void tripolar_vy_sync_valid (MultiFab& v, Geometry const& geom);
void tripolar_vy_fill_ghost (MultiFab& v, Geometry const& geom);
void tripolar_vx_fill_ghost (MultiFab& u, Geometry const& geom);

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int nx = 128;
        int ny = 128;
        int nz = 8;
        int max_grid_size = 32;
        Box domain(IntVect(0), IntVect(nx-1,ny-1,nz-1));
        Array<int,3> is_per{1,0,0}; // periodic in x-direction
        Geometry geom(domain, RealBox({0.,0.,0.}, {1.,1.,1.}), 0, is_per);
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};

        // y-velocity
        MultiFab v(amrex::convert(ba,IntVect(0,1,0)), dm, 1, 1);

        // Fill with random data
        FillRandom(v, 0, v.nComp());

        // Because the data are random, the shared faces do not have
        // consistent data. So we can fix it with OverrideSync that
        // synchronizes data on shared faces/nodes.
        v.OverrideSync(geom.periodicity());

        // For v (i.e., y-velocity), we can also synchronize its data at the
        // valid face by overriding the upper half. We will handle ghost
        // cells later.
        tripolar_vy_sync_valid(v, geom);

        // Let's fill the ghost faces at y-hi boundary of v.
        tripolar_vy_fill_ghost(v, geom);

        // Regular FillBoundary
        v.FillBoundary(geom.periodicity());

        // Verification
        {
            Box test_box = amrex::bdryHi(domain,1);
            test_box.grow(1, 1); // include ghost in y-hi-direction.
            test_box.grow(0, 1); // include ghost in x-direction
            MultiFab tmp(BoxArray(test_box), DistributionMapping(Vector<int>{0}), 1, 0);
            tmp.ParallelCopy(v, 0, 0, 1, IntVect(1,1,0), IntVect(0)); // Copy data to rank 0
            if (ParallelDescriptor::MyProc() == 0) {
                auto const& a = tmp.array(0);
                ParallelFor(Box(a), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // What are we supposed to do at i <= 0 and i >= nx-?
                    int ii;
                    if (i < 0) {
                        ii = (nx-1) - (i+nx);
                    } else if (i >= nx) {
                        ii = (nx-1) - (i-nx);
                    } else {
                        ii = (nx-1) - i;
                    }
                    int jj = 2*ny - j;
                    if (a(i,j,k) != -a(ii,jj,k)) {
                        amrex::AllPrint() << "Failed! v(" << i << "," << j << "," << k << ") = " << a(i,j,k)
                                          << " v(" << ii << "," << jj << "," << k << ") = " << a(ii,jj,k)
                                          << "\n";
                    }
                });
            }
        }

        // x-velocity
        MultiFab u(amrex::convert(ba,IntVect(1,0,0)), dm, 1, 1);

        // Fill with random data
        FillRandom(u, 0, u.nComp());

        // Because the data are random, the shared faces do not have
        // consistent data. So we can fix it with OverrideSync that
        // synchronizes data on shared faces/nodes.
        u.OverrideSync(geom.periodicity());

        tripolar_vx_fill_ghost(u, geom);

        // Verification
        {
            Box test_box = amrex::adjCellHi(domain,1);
            test_box.convert(IntVect(1,0,0));
            test_box.growLo(1, 1); // include one layer of valid cells in y-direcion
            MultiFab tmp(BoxArray(test_box), DistributionMapping(Vector<int>{0}), 1, 0);
            tmp.ParallelCopy(u, 0, 0, 1, IntVect(0,1,0), IntVect(0)); // Copy data to rank 0
            if (ParallelDescriptor::MyProc() == 0) {
                auto const& a = tmp.array(0);
                ParallelFor(Box(a), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // What are we supposed to do at i == 0, nx/2 and nx?
                    int ii = nx - i;
                    int jj = (2*ny-1) - j;
                    if (a(i,j,k) != -a(ii,jj,k)) {
                        amrex::AllPrint() << "Failed! u(" << i << "," << j << "," << k << ") = " << a(i,j,k)
                                          << " u(" << ii << "," << jj << "," << k << ") = " << a(ii,jj,k)
                                          << "\n";
                    }
                });
            }
        }
    }
    amrex::Finalize();
}

void tripolar_vy_sync_valid (MultiFab& v, Geometry const& geom)
{
    AMREX_ALWAYS_ASSERT(v.ixType().toIntVect() == IntVect(0,1,0));

    struct IndexMapping {
        int nx;

        [[nodiscard]] Dim3 operator() (Dim3 i) const noexcept {
            return {nx-1-i.x, i.y, i.z};
        }

        [[nodiscard]] Dim3 Inverse (Dim3 i) const noexcept {
            return {nx-1-i.x, i.y, i.z};
        }

        [[nodiscard]] constexpr IndexType operator() (IndexType it) const noexcept {
            return it;
        }

        static constexpr IndexType Inverse (IndexType it) { return it; }
    };

    struct Flip {
        Real operator()(Array4<const Real> const& array, Dim3 i, int comp) const noexcept {
            return -array(i.x, i.y, i.z, comp);
        }
    };

    Box const& domain = geom.Domain();

    Box yhibox = amrex::bdryHi(domain, 1);
    int nx = domain.length(0);
    AMREX_ALWAYS_ASSERT(nx%2 == 0);
    yhibox.setSmall(0, nx/2);

    IndexMapping dtos{nx};
    Flip packing{};

    MultiBlockCommMetaData cmd(v, yhibox, v, IntVect(0), dtos);

    ParallelCopy(v, v, cmd, 0, 0, 1, dtos, packing);
}

void tripolar_vy_fill_ghost (MultiFab& v, Geometry const& geom)
{
    AMREX_ALWAYS_ASSERT(v.ixType().toIntVect() == IntVect(0,1,0));

    struct IndexMapping {
        int nx, ny;

        [[nodiscard]] Dim3 operator() (Dim3 i) const noexcept {
            return {nx-1-i.x, 2*ny-i.y, i.z};
        }

        [[nodiscard]] Dim3 Inverse (Dim3 i) const noexcept {
            return this->operator()(i);
        }

        [[nodiscard]] constexpr IndexType operator() (IndexType it) const noexcept {
            return it;
        }

        static constexpr IndexType Inverse (IndexType it) { return it; }
    };

    struct Flip {
        Real operator()(Array4<const Real> const& array, Dim3 i, int comp) const noexcept {
            return -array(i.x, i.y, i.z, comp);
        }
    };

    Box const& domain = geom.Domain();

    Box yhibox = amrex::shift(amrex::bdryHi(domain, 1), 1, 1);
    int nx = domain.length(0);
    int ny = domain.length(1);
    AMREX_ALWAYS_ASSERT(nx%2 == 0);

    IndexMapping dtos{nx,ny};
    Flip packing{};

    MultiBlockCommMetaData cmd(v, yhibox, v, IntVect(1), dtos);

    v.FillBoundary(geom.periodicity());
    ParallelCopy(v, v, cmd, 0, 0, 1, dtos, packing);
    v.EnforcePeriodicity(geom.periodicity());
}

void tripolar_vx_fill_ghost (MultiFab& u, Geometry const& geom)
{
    AMREX_ALWAYS_ASSERT(u.ixType().toIntVect() == IntVect(1,0,0));

    struct IndexMapping {
        int nx;

        [[nodiscard]] Dim3 operator() (Dim3 i) const noexcept {
            return {nx-i.x, i.y-1, i.z};
        }

        [[nodiscard]] Dim3 Inverse (Dim3 i) const noexcept {
            return {nx-i.x, i.y+1, i.z};
        }

        [[nodiscard]] constexpr IndexType operator() (IndexType it) const noexcept {
            return it;
        }

        static constexpr IndexType Inverse (IndexType it) { return it; }
    };

    struct Flip {
        Real operator()(Array4<const Real> const& array, Dim3 i, int comp) const noexcept {
            return -array(i.x, i.y, i.z, comp);
        }
    };

    Box const& domain = geom.Domain();

    Box yhibox = amrex::adjCellHi(domain, 1);
    yhibox.convert(IntVect(1,0,0));
    int nx = domain.length(0);
    AMREX_ALWAYS_ASSERT(nx%2 == 0);

    IndexMapping dtos{nx};
    Flip packing{};

    MultiBlockCommMetaData cmd(u, yhibox, u, IntVect(1), dtos);

    u.FillBoundary(geom.periodicity());
    ParallelCopy(u, u, cmd, 0, 0, 1, dtos, packing);
    u.EnforcePeriodicity(geom.periodicity());
}
