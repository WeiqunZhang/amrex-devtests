This is an example of how one can build kokkos and amrex. Note that AMReX
needs `Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE`

* Compile Kokkos

    $ cd kokkos
    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_INSTALL_PREFIX=/home/wqzhang/opt/kokkos -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON -DKokkos_ARCH_PASCAL60=ON
    $ make -j8
    $ make install
    
* Compile amrex

    $ cd amrex
    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_INSTALL_PREFIX=/home/wqzhang/opt/amrex -DAMReX_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=60 -DAMReX_MPI=OFF
    $ make -j8
    $ make install

* Compile this test code

    $ mkdir build
    $ cd build
    $ cmake .. -DENABLE_CUDA=ON -DAMReX_ROOT=/home/wqzhang/opt/amrex -DKokkos_ROOT=/home/wqzhang/opt/kokkos -DCMAKE_CUDA_ARCHITECTURES=60
    $ make -j8
