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

DEFINES += -DSTREAM_ARRAY_SIZE=134217728 -DNTIMES=100

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
