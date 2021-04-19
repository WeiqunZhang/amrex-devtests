AMREX_HOME ?= ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_CUDA  = TRUE
USE_HIP   = FALSE
USE_DPCPP = FALSE

USE_MPI   = FALSE
USE_OMP   = FALSE

BL_NO_FORT = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
