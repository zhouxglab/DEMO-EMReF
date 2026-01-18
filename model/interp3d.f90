! interp3d.f90 
! Interpolation of EM density grids

! Copyright (C) 2024  Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.

module interp3d
real(kind=4), allocatable :: mapout(:, :, :)
integer pextx, pexty, pextz
contains
    subroutine linear(mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx)
    implicit none
    real zpix, ypix, xpix, apix, shiftz, shifty, shiftx
    integer nx, ny, nz
    real mapin(nz, ny, nx)
    integer indx, indy, indz
    real xpos, ypos, zpos, gx, gy, gz, a, b, c
    integer x0, y0, z0, x1, y1, z1

    pextx = floor(xpix * (nx - 1) / apix) + 1
    pexty = floor(ypix * (ny - 1) / apix) + 1
    pextz = floor(zpix * (nz - 1) / apix) + 1
    allocate(mapout(pextz, pexty, pextx))
    do indz = 1, pextz
        do indy = 1, pexty
            do indx = 1, pextx
                xpos = (indx - 1) * apix + shiftx
                ypos = (indy - 1) * apix + shifty
                zpos = (indz - 1) * apix + shiftz
                gx = xpos / xpix + 1
                gy = ypos / ypix + 1
                gz = zpos / zpix + 1
                x0 = floor(gx)
                y0 = floor(gy)
                z0 = floor(gz)
                x1 = x0 + 1
                y1 = y0 + 1
                z1 = z0 + 1
                if(x0 >= 1 .and. x1 <= nx &
             .and. y0 >= 1 .and. y1 <= ny &
             .and. z0 >= 1 .and. z1 <= nz) then
                    a = gx - x0
                    b = gy - y0
                    c = gz - z0
           mapout(indz, indy, indx)=a    *b    *c    *mapin(z1,y1,x1) &
                                   +(1-a)*b    *c    *mapin(z1,y1,x0) &
                                   +a    *(1-b)*c    *mapin(z1,y0,x1) &
                                   +a    *b    *(1-c)*mapin(z0,y1,x1) &
                                   +a    *(1-b)*(1-c)*mapin(z0,y0,x1) &
                                   +(1-a)*b    *(1-c)*mapin(z0,y1,x0) &
                                   +(1-a)*(1-b)*c    *mapin(z1,y0,x0) &
                                   +(1-a)*(1-b)*(1-c)*mapin(z0,y0,x0)
                end if
            end do 
        end do 
    end do

    return
    end subroutine

    subroutine cubic(mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx)
    implicit none
    real zpix, ypix, xpix, apix, shiftz, shifty, shiftx
    integer nz, ny, nx
    real(kind=4) :: mapin(nz, ny, nx)

    integer indz, indy, indx
    real gz, gy, gx, wz(4), wy(4), wx(4)
    integer intz, inty, intx, i, j, k

    pextz = floor(zpix * (nz - 1) / apix) + 1
    pexty = floor(ypix * (ny - 1) / apix) + 1
    pextx = floor(xpix * (nx - 1) / apix) + 1

    allocate(mapout(pextz, pexty, pextx))
    mapout(:, :, :) = 0.0

    do indz = 1, pextz
        do indy = 1, pexty
            do indx = 1, pextx

                gz = ((indz - 1) * apix + shiftz) / zpix + 1
                gy = ((indy - 1) * apix + shifty) / ypix + 1
                gx = ((indx - 1) * apix + shiftx) / xpix + 1

                intz = floor(gz)
                inty = floor(gy)
                intx = floor(gx)

                if (intz >= 1 .and. intz + 1 <= nz &
              .and. inty >= 1 .and. inty + 1 <= ny &
              .and. intx >= 1 .and. intx + 1 <= nx) then
                    call get_w(gz, wz)
                    call get_w(gy, wy)
                    call get_w(gx, wx)
                    do i = 1, 4 ! shift z
                        do j = 1, 4 ! shift y
                            do k = 1, 4 ! shift x
                                if (intz + i - 2 >= 1 .and. intz + i - 2 <= nz &
                              .and. inty + j - 2 >= 1 .and. inty + j - 2 <= ny &
                              .and. intx + k - 2 >= 1 .and. intx + k - 2 <= nx) then
                                    mapout(indz, indy, indx) = mapout(indz, indy, indx) &
                                 +  mapin(intz + i - 2, inty + j - 2, intx + k - 2) * wz(i) * wy(j) * wx(k)
                                end if
                            end do
                        end do
                    end do
                end if
            end do
        end do
    end do

    return
    end subroutine cubic

    subroutine inverse_cubic(mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx)
    implicit none
    real zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx
    integer nz, ny, nx
    real(kind=4) :: mapin(nz, ny, nx)

    integer indz, indy, indx
    real gz, gy, gx, wz(4), wy(4), wx(4)
    integer intz, inty, intx, i, j, k

    pextz = ceiling(zpix * (nz - 1) / zpix_o) + 1
    pexty = ceiling(ypix * (ny - 1) / ypix_o) + 1
    pextx = ceiling(xpix * (nx - 1) / xpix_o) + 1

    allocate(mapout(pextz, pexty, pextx))
    mapout(:, :, :) = 0.0

    do indz = 1, pextz
        do indy = 1, pexty
            do indx = 1, pextx

                gz = ((indz - 1) * zpix_o + shiftz) / zpix + 1
                gy = ((indy - 1) * ypix_o + shifty) / ypix + 1
                gx = ((indx - 1) * xpix_o + shiftx) / xpix + 1

                intz = floor(gz)
                inty = floor(gy)
                intx = floor(gx)

                if (intz >= 1 .and. intz + 1 <= nz &
              .and. inty >= 1 .and. inty + 1 <= ny &
              .and. intx >= 1 .and. intx + 1 <= nx) then
                    call get_w(gz, wz)
                    call get_w(gy, wy)
                    call get_w(gx, wx)
                    do i = 1, 4 ! shift z
                        do j = 1, 4 ! shift y
                            do k = 1, 4 ! shift x
                                if (intz + i - 2 >= 1 .and. intz + i - 2 <= nz &
                              .and. inty + j - 2 >= 1 .and. inty + j - 2 <= ny &
                              .and. intx + k - 2 >= 1 .and. intx + k - 2 <= nx) then
                                    mapout(indz, indy, indx) = mapout(indz, indy, indx) &
                                 +  mapin(intz + i - 2, inty + j - 2, intx + k - 2) * wz(i) * wy(j) * wx(k)
                                end if
                            end do
                        end do
                    end do
                end if
            end do
        end do
    end do

    return
    end subroutine inverse_cubic

    subroutine del_mapout
    if (allocated(mapout)) then
        deallocate(mapout)
    end if
    return 
    end subroutine

    subroutine get_w(x, w)
    implicit none
    real, parameter :: a = -0.5
    real, intent(in) :: x
    real :: d1, d2, d3, d4
    integer :: intx
    real, intent(out) :: w(4)
    w(:) = 0.0
    intx = floor(x)
    d1 = 1.0 + (x - intx)
    d2 = d1 - 1.0
    d3 = 1.0 - d2
    d4 = d3 + 1.0
    w(1) = a*abs(d1**3) - 5*a*d1**2 + 8*a*abs(d1) - 4*a
    w(2) = (a+2)*abs(d2**3) - (a+3)*d2**2 + 1
    w(3) = (a+2)*abs(d3**3) - (a+3)*d3**2 + 1
    w(4) = a*abs(d4**3) - 5*a*d4**2 + 8*a*abs(d4) - 4*a 
    return
    end subroutine get_w
end module  
