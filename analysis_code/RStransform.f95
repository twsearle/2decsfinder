!A subroutine for transforming a complex array of numbers to real space
      SUBROUTINE RStransform(inarr, oarr, N, M, numXs, numYs, kx)
          implicit none
          integer :: N 
          integer :: M 
          integer :: numXs 
          integer :: numYs 
          integer :: ndx
          integer :: mdx
          integer :: xIndx
          integer :: yIndx
          real :: kx
          real :: xpos
          real :: ypos
          real :: pi
          real :: tmp3
          double complex :: tmp
          double complex :: tmp2
          double complex :: II = (0.0,1.0)
          double complex ,dimension((2*N+1)*M) :: inarr
          double complex ,dimension(numXs*numYs) :: oarr
          !print *, numXs
          !print *, numYs
          !print *, N
          !print *, M

          pi = acos(-1.0)

          do xIndx = 0,numXs-1
            do yIndx = 1,numYs
            xpos = 4.0*pi/kx * (real(xIndx)/real(numXs-1.0))
            ypos = 2.0*real(yIndx-1)/(real(numYs)-1.0)-1.0
              do ndx = 0,(2*N)
                tmp2 = exp(II*cmplx(ndx-N)*cmplx(kx*xpos))
                do mdx = 1,M
                  tmp3 = cos((mdx-1)*acos(ypos))
                  tmp = inarr(ndx*M+mdx)*tmp2*cmplx(tmp3)
                  oarr(numYs*xIndx+yIndx) = oarr(numYs*xIndx+yIndx)+tmp
                end do
              end do
            end do
          end do

          END
