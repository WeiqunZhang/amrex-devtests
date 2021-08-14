AMREX_HOME ?= ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_DPCPP = FALSE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

ifeq ($(USE_HIP),TRUE)
  XTRA_CXXFLAGS = -ffast-math -fhonor-nans -fhonor-infinites -fdenormal-fp-math=ieee -fcuda-flush-denormals-to-zero -munsafe-fp-atomics
  XTRA_CFLAGS   = -ffast-math -fhonor-nans -fhonor-infinites -fdenormal-fp-math=ieee -fcuda-flush-denormals-to-zero -munsafe-fp-atomics
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
