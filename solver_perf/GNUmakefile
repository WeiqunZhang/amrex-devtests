DEBUG = FALSE
USE_MPI  = TRUE
USE_OMP  = FALSE
COMP = gnu
DIM = 3

USE_CUDA  = FALSE
USE_DPCPP = FALSE
USE_HIP   = FALSE

AMREX_HOME = ../../amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs 	:= Base Boundary LinearSolvers/MLMG

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

