
subroutine fort_stream_copy (a, alo, ahi, c, clo, chi) bind(c)
  use amrex_fort_module, only : amrex_real
  implicit none
  integer, intent(in) :: alo(3), ahi(3)
  integer, intent(in) :: clo(3), chi(3)
  real(amrex_real) :: a(alo(1):ahi(1),alo(2):ahi(2),alo(3):ahi(3))
  real(amrex_real) :: c(clo(1):chi(1),clo(2):chi(2),clo(3):chi(3))

  integer :: i, j, k

  do       k = clo(3), chi(3)
     do    j = clo(2), chi(2)
        do i = clo(1), chi(1)
           c(i,j,k) = a(i,j,k)
        end do
     end do
  end do

end subroutine fort_stream_copy

subroutine fort_stream_scale (b, blo, bhi, c, clo, chi, scalar) bind(c)
  use amrex_fort_module, only : amrex_real
  implicit none
  integer, intent(in) :: blo(3), bhi(3)
  integer, intent(in) :: clo(3), chi(3)
  real(amrex_real) :: b(blo(1):bhi(1),blo(2):bhi(2),blo(3):bhi(3))
  real(amrex_real) :: c(clo(1):chi(1),clo(2):chi(2),clo(3):chi(3))
  real(amrex_real), value :: scalar

  integer :: i, j, k

  do       k = clo(3), chi(3)
     do    j = clo(2), chi(2)
        do i = clo(1), chi(1)
           b(i,j,k) = scalar * c(i,j,k)
        end do
     end do
  end do

end subroutine fort_stream_scale

subroutine fort_stream_add (a, alo, ahi, b, blo, bhi, c, clo, chi) bind(c)
  use amrex_fort_module, only : amrex_real
  implicit none
  integer, intent(in) :: alo(3), ahi(3)
  integer, intent(in) :: blo(3), bhi(3)
  integer, intent(in) :: clo(3), chi(3)
  real(amrex_real) :: a(alo(1):ahi(1),alo(2):ahi(2),alo(3):ahi(3))
  real(amrex_real) :: b(blo(1):bhi(1),blo(2):bhi(2),blo(3):bhi(3))
  real(amrex_real) :: c(clo(1):chi(1),clo(2):chi(2),clo(3):chi(3))

  integer :: i, j, k

  do       k = clo(3), chi(3)
     do    j = clo(2), chi(2)
        do i = clo(1), chi(1)
           c(i,j,k) = a(i,j,k) + b(i,j,k)
        end do
     end do
  end do

end subroutine fort_stream_add

subroutine fort_stream_triad (a, alo, ahi, b, blo, bhi, c, clo, chi, scalar) bind(c)
  use amrex_fort_module, only : amrex_real
  implicit none
  integer, intent(in) :: alo(3), ahi(3)
  integer, intent(in) :: blo(3), bhi(3)
  integer, intent(in) :: clo(3), chi(3)
  real(amrex_real) :: a(alo(1):ahi(1),alo(2):ahi(2),alo(3):ahi(3))
  real(amrex_real) :: b(blo(1):bhi(1),blo(2):bhi(2),blo(3):bhi(3))
  real(amrex_real) :: c(clo(1):chi(1),clo(2):chi(2),clo(3):chi(3))
  real(amrex_real), value :: scalar

  integer :: i, j, k

  do       k = clo(3), chi(3)
     do    j = clo(2), chi(2)
        do i = clo(1), chi(1)
           a(i,j,k) = b(i,j,k) + scalar * c(i,j,k)
        end do
     end do
  end do

end subroutine fort_stream_triad

