AMREX_HOME ?= ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = TRUE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

SYCL_WORKAROUND = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(SYCL_WORKAROUND),FALSE)
  DEFINES += -DAMREX_NO_SYCL_REDUCE_WORKAROUND
endif

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
