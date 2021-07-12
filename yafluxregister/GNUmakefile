
PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = TRUE
DEBUG      = FALSE

DIM        = 2
#DIM       = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE 
USE_CUDA   = FALSE

BL_NO_FORT = TRUE

AMREX_HOME = ../../amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
