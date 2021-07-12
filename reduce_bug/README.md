
AMReX has a workaround for an DPC++ issue in AMReX's reduction functions at
`amrex/Src/Base/AMReX_Reduce.H`.  We use a two passes algorithm.  The second
pass is done in `ReduceOps::value()` starting at around line 514.  We have
to do the second pass twice in order to get correct results consistently.
See line 514 of `amrex/Src/Base/AMReX_Reduce.H` for more detail.  The
workaround is enabled by default in amrex, but is disabled in this test.  To
enable it for this test, `make` with `DPCPP_WORKAROUND=TRUE`.
