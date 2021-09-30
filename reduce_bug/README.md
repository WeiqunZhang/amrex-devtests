
AMReX has a workaround for a DPC++ issue in AMReX's reduction functions at
`amrex/Src/Base/AMReX_Reduce.H`.  Se line 514 of
`amrex/Src/Base/AMReX_Reduce.H` for more detail.  It seems that the issue is
that writing to pinned memory on device often fails.  The workaround is
enabled by default in amrex, but is disabled in this test.  To enable it for
this test, `make` with `DPCPP_WORKAROUND=TRUE`.
