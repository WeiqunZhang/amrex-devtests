AMREX_HOME ?= ../../amrex

DEBUG	= TRUE

DIM	= 2

COMP    = gcc

USE_OWN_FILE_STREAM = TRUE

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
