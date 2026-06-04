! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
module mod_opposition
use mod_types
use mpi
use mod_common_mpi, only: canscomm
!@acc use cudecomp
implicit none
public
!
!
! opposition control read input and routines
!
!
!integer , protected :: z_sense, mysize
!real(rp), protected :: alpha_opp
integer , protected :: z_sense, n_walls
real(rp), protected :: alpha_opp
! Needed for routines
real(rp), allocatable, dimension(:,:,:) :: var_opp

contains
subroutine read_opposition(myid)
    implicit none
    integer, intent(in) :: myid
    integer :: iunit, ierr
    namelist /opp/ z_sense, n_walls, alpha_opp
    
    ! Initialize with invalid values to check if they're properly read
    z_sense = -999
    n_walls = -999
    alpha_opp = -999.0_rp
    
    open(newunit=iunit, file='opp.nml', status='old', action='read', iostat=ierr)
    if (ierr /= 0) then
        if (myid == 0) then
            print *, 'Error opening opp.nml file, iostat = ', ierr
            print *, 'Aborting...'
        end if
        call MPI_ABORT(canscomm, 1, ierr)
    end if
    
    read(iunit, nml=opp, iostat=ierr)
    if (ierr /= 0) then
        if (myid == 0) then
            print *, 'Error reading namelist from opp.nml, iostat = ', ierr
            print *, 'Aborting...'
        end if
        call MPI_ABORT(canscomm, 1, ierr)
    end if
    
    close(iunit)
    
    ! Verify values were properly read
    if (z_sense == -999 .or. alpha_opp == -999.0_rp .or. n_walls == -999) then
        if (myid == 0) then
            print *, 'Error: Some variables were not properly read from namelist'
            print *, 'Aborting...'
        end if
        call MPI_ABORT(canscomm, 1, ierr)
    end if
    
    if (myid == 0) then
        print *, 'Successfully read from opp.nml:'
        print *, '  alpha_opp = ', alpha_opp
        print *, '  z_sense = ', z_sense
        print *, '  n_walls = ', n_walls
    end if
end subroutine read_opposition

subroutine opposition(n,alpha_opp,z,var_opp,w)
  implicit none
  integer , intent(in),    dimension(3)     :: n
  real(rp), intent(in)                      :: alpha_opp
  integer , intent(in)                      :: z
  real(rp), intent(in),    dimension(:,:,0:)   :: var_opp
  real(rp), intent(inout), dimension(0:,0:,0:) :: w
  integer  :: i,j,k_wall

  k_wall = z * n(3)
  ! async(1) to serialize with set_bc (which uses async(1) to zero w at the
  ! wall just before this call inside bounduvw); without this the null-queue
  ! kernel can race with set_bc and the opposition write gets clobbered.
  !$acc parallel loop collapse(2) present(w, var_opp) async(1)
  do j=1,n(2)
    do i=1,n(1)
      w(i,j,k_wall) = alpha_opp * var_opp(i,j,k_wall)
    end do
  end do
  !$acc wait(1)

end subroutine opposition

subroutine update_opp(n, n_walls, w, var_opp, myid)
  use mod_drl, only: var_opp_marl
  implicit none
  integer, intent(in), dimension(3) :: n
  integer, intent(in) :: n_walls
  real(rp), intent(in), dimension(0:,0:,0:) :: w
  real(rp), intent(inout), dimension(:,:,0:) :: var_opp
  integer, intent(in) :: myid
  integer :: i,j

  ! Python supplies the wall BC directly via var_opp_marl (refreshed from
  ! host after each MPI_RECV in mod_drl). Whatever Python sends — raw
  ! +/-1 bang-bang, opposition = -(w-<w>), a learned policy output, or
  ! zero — is written verbatim to the wall. No native opposition is
  ! added here.
  !$acc parallel loop collapse(2) present(var_opp, var_opp_marl)
  do j=1,n(2)
    do i=1,n(1)
      var_opp(i,j,0) = var_opp_marl(i,j)
    end do
  end do

  if (n_walls.eq.2) then
    !$acc parallel loop collapse(2) present(var_opp, var_opp_marl)
    do j=1,n(2)
      do i=1,n(1)
        var_opp(i,j,1) = var_opp_marl(i,j)
      end do
    end do
  end if
end subroutine update_opp

end module
