AMREX_HOME = ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gnu

USE_CUDA  = TRUE
USE_HIP   = FALSE
USE_SYCL  = FALSE

USE_MPI   = FALSE
USE_OMP   = FALSE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(USE_CUDA),TRUE)
  LIBRARIES += -lcufft
endif

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
