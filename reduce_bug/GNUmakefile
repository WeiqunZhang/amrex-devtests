AMREX_HOME ?= ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_DPCPP = TRUE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

DPCPP_WORKAROUND = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(DPCPP_WORKAROUND),FALSE)
  DEFINES += -DAMREX_NO_DPCPP_REDUCE_WORKAROUND
endif

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
