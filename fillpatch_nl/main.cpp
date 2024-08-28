#include <AMReX.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    {
        ParmParse pp("amrex");
        pp.add("fpe_trap_invalid",1);
        pp.add("fpe_trap_overflow",1);
        pp.add("fpe_trap_zero",1);
    }
    amrex::Initialize(argc,argv);
    {
        Box dom0(IntVect(0), IntVect(511,383,7));
        RealBox rb({-10.0, -12.0, -0.0}, {22.0, 12.0, 0.5});
        Array<int,3> is_periodic{0,0,1};

        int const nlevels = 5;

        Vector<Geometry> geom;
        geom.emplace_back(dom0, rb, CoordSys::cartesian, is_periodic);
        for (int ilev = 1; ilev < nlevels; ++ilev) {
            geom.push_back(amrex::refine(geom.back(), 2));
        }

        Long ntotcells = 0;

        Vector<BoxArray> grids;
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            amrex::Print() << "BoxArray on Level " << ilev << "\n";
            Vector<char> buf;
            ParallelDescriptor::ReadAndBcastFile("boxarray"+std::to_string(ilev)+".txt", buf);
            std::string sbuf(buf.dataPtr());
            std::istringstream is(sbuf, std::istringstream::in);
            BoxArray ba;
            amrex::readBoxArray(ba, is);
            grids.emplace_back(std::move(ba));
            ntotcells += grids[ilev].numPts();
            amrex::Print() << "    numPts: " << grids[ilev].numPts() << "\n"
                           << "    bounding box: " << grids[ilev].minimalBox() << "\n";
            if (ilev > 0) {
                if (grids[ilev-1].contains(amrex::coarsen(grids[ilev],2))) {
                    amrex::Print() << "    BoxArray " << ilev-1 << " contains BoxArray " << ilev << "\n";
                } else {
                    amrex::Print() << "    BoxArray " << ilev-1 << " does NOT contain BoxArray " << ilev << "\n";
                }
            }
        }

        BoxArray bamf;
        {
            amrex::Print() << "BoxArray of mf\n";
            Vector<char> buf;
            ParallelDescriptor::ReadAndBcastFile("boxarraymf.txt", buf);
            std::string sbuf(buf.dataPtr());
            std::istringstream is(sbuf, std::istringstream::in);
            amrex::readBoxArray(bamf, is);
            amrex::Print() << "    numPts: " << bamf.numPts() << "\n"
                           << "    bounding box: " << bamf.minimalBox() << "\n";
        }

        amrex::Print() << "Total number of cells: " << ntotcells << "\n";

        Vector<MultiFab> leveldata(nlevels);
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            leveldata[ilev].define(grids[ilev], DistributionMapping(grids[ilev]), 1, 0);
            auto const& ma = leveldata[ilev].arrays();
            auto const& problo = geom[ilev].ProbLoArray();
            auto const& dx = geom[ilev].CellSizeArray();
            ParallelFor(leveldata[ilev],
                        [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
                        {
                            ma[b](i,j,k) = problo[0] + (i+0.5)*dx[0];
                        });
        }

        MultiFab mf(bamf, DistributionMapping(bamf), 1, 0);

        Vector<Vector<MultiFab*>> smf(nlevels);
        Vector<Vector<Real>> st(nlevels);
        
        Vector<PhysBCFunct<GpuBndryFuncFab<FabFillNoOp>>> bc;
        Vector<IntVect> ratio(nlevels-1, IntVect(2));
        Vector<BCRec> bcr(nlevels);
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            smf[ilev].push_back(&leveldata[ilev]);
            st[ilev].push_back(0.0);
            bcr[ilev] = BCRec(BCType::foextrap, BCType::foextrap, BCType::int_dir,
                              BCType::foextrap, BCType::foextrap, BCType::int_dir);
            bc.emplace_back(geom[ilev], Vector<BCRec>{bcr[ilev]}, FabFillNoOp{});
        }

        int levelmf = 4;
        FillPatchNLevels(mf, levelmf, IntVect(0), 0.0, smf, st, 0, 0, 1,
                         geom, bc, 0, ratio, &cell_cons_interp, bcr, 0);

        if (mf.is_finite()) {
            amrex::Print() << "mf.min & max: " << mf.min(0) << " " << mf.max(0) << "\n";
            amrex::Print() << "SUCCESS\n";
        } else {
            amrex::Print() << "FAIL\n";
        }
    }
    amrex::Finalize();
}
