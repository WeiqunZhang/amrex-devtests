# Notes on Filling Boundaries in Spherical Coordinates

Here the spherical coordinates are (theta, phi, r) and we will denote them
as (x,y,z). There are 26 cases.

1. xlo-face: i < 0 and 0 <= j < ny and 0 <= k < nz
   s(i,j,k) = s(-1-i, (j+ny/2)%ny, k)
  
2. xhi-face: i >= nx and 0 <= j < ny and 0 <= k < nz
   s(i,j,k) = s(2*nx-1-i, (j+ny/2)%ny, k)
  
3. ylo-face: 0 <= i < nx and j < 0 and 0 <= k < nz
   s(i,j,k) = s(i, j+ny, k)
  
4. yhi-face: 0 <= i < nx and j >= ny and 0 <= k < nz
   s(i,j,k) = s(i, j-ny, k)
  
5. zlo-face: 0 <= i < nx and 0 <= j < ny and k < 0
   s(i,j,k) = s(nx-1-i, (j+ny/2)%ny, -1-k)
  
6. DONE zhi-face: 0 <= i < nx and 0 <= j < ny and k >= nz
   extrap 
  
7. xlo-ylo-edge: i < 0 and 0 < j and 0 <= k < nz
   s(i,j,k) = s(-1-i, (j+ny/2)%ny, k)

8. xhi-ylo-edge: i >= nx and 0 < j and 0 <= k < nz
   s(i,j,k) = s(2*nx-1-i, (j+ny/2)%ny, k)
  
9. xlo-yhi-edge: i < 0 and j >= ny and 0 <= k < nz
   s(i,j,k) = s(-1-i, (j+ny/2)%ny, k)

10. xhi-yhi-edge: i >= nx and j >= ny and 0 <= k < nz
    s(i,j,k) = s(2*nx-1-i, (j+ny/2)%ny, k)
  
11. xlo-zlo-edge: i < 0 and 0 <= y < ny and k < 0
    N/A
  
12. xhi-zlo-edge: i >= nx and 0 <= y < ny and k < 0
    N/A
  
13. DONE xlo-zhi-edge: i < 0 and 0 <= y < ny and k >= nz
    extrap
  
14. DONE xhi-zhi-edge: i >= nx and 0 <= y < ny and k >= nz
    extrap
  
15. ylo-zlo-edge: 0 <= i < nx and j < 0 and k < 0
    s(i,j,k) = s(nx-1-i, (j+ny/2)%ny, -1-k)

16. yhi-zlo-dedge: 0 <= i < nx and j >= ny and k < 0
    s(i,j,k) = s(nx-1-i, (j+ny/2)%ny, -1-k)

17. DONE ylo-zhi-edge: 0 <= i < nx and j < 0 and k >= nz
    extrap

18. DONE yhi-zhi-edge: 0 <= i < nx and j >= ny and k >= nz
    extrap

19. xlo-ylo-zlo-corner: i < 0 and j < 0 and k < 0
    N/A

20. xhi-ylo-zlo-corner: i >= nx and j < 0 and k < 0
    N/A

21. xlo-yhi-zlo-corner: i < 0 and j >= ny and k < 0
    N/A

22. xhi-yhi-zlo-corner: i >= nx and j >= ny and k < 0
    N/A

23. DONE xlo-ylo-zhi-corner: i < 0 and j < 0 and k >= nz
    extrap

24. DONE xhi-ylo-zhi-corner: i >= nx and j < 0 and k >= nz
    extrap

25. DONE xlo-yhi-zhi-corner: i < 0 and j >= ny and k >= nz
    extrap

26. DONE xhi-yhi-zhi-corner: i >= nx and j >= ny and k >= nz
    extrap
