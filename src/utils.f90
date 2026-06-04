! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
module mod_utils
  use mod_common_mpi, only:canscomm
  implicit none
  public 
  !@acc public device_memory_footprint
contains
  subroutine bulk_mean(n,grid_vol_ratio,p,mean)
    !
    ! compute the mean value of an observable over the entire domain
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: p
    real(rp), intent(out) :: mean
    integer :: i,j,k
    integer :: ierr
    mean = 0.
    !$acc data copy(mean) async(1)
    !$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)  REDUCTION(+:mean)
    do k=1,n(3)
      do j=1,n(2)
        do i=1,n(1)
          mean = mean + p(i,j,k)*grid_vol_ratio(k)
        end do
      end do
    end do
    !$acc end data
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean

  subroutine bulk_mean_mod(n,grid_vol_ratio,p,ind,mean)
    !
    ! compute the mean value of an observable over a box
    ! of dimensions (Lx, Ly, z(ind))
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: p
    integer , intent(in)  :: ind
    real(rp), intent(out) :: mean
    integer :: i,j,k
    integer :: ierr
    mean = 0.
    !$acc data copy(mean) async(1)
    !$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)  REDUCTION(+:mean)
    do k=1,ind
      do j=1,n(2)
        do i=1,n(1)
          mean = mean + p(i,j,k)*grid_vol_ratio(k)
        end do
      end do
    end do
    !$acc end data
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean_mod

  subroutine bulk_mean_2d(n,grid_vol_ratio,u,w,ind,mean1,mean2)
    !
    ! compute the mean value of an observable over a box
    ! of dimensions (Lx, Ly, z(ind))
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: u,w
    integer , intent(in)  :: ind
    real(rp), intent(out) :: mean1, mean2
    integer :: i,j,k
    integer :: ierr
    real :: ui, wi
    mean1 = 0.
    mean2 = 0.
    !!$acc data copy(mean) async(1)  ! disabled: mean not declared here
    !!$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(2) DEFAULT(shared)  REDUCTION(+:mean1,mean2)
      do j=1,n(2)
        do i=1,n(1)
          ui = u(i,j,ind)*grid_vol_ratio(ind)
          wi = 0.5*(w(i,j,ind)*grid_vol_ratio(ind) + w(i,j,ind+1)*grid_vol_ratio(ind+1))
          if ((ui.gt.(0.)).and.(wi.lt.(0.))) then
             !mean1 = mean1 + abs(ui*wi)
             mean1 = mean1 + 1.
          else
             !mean2 = mean2 + abs(ui*wi)
             mean2 = mean2 + 1.
          end if
        end do
      end do
    !!$acc end data
    !!$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean1,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean2,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean_2d

  subroutine bulk_mean_mod_sq(n,grid_vol_ratio,p,ind,mean)
    !
    ! compute the mean value of an observable over a box
    ! of dimensions (Lx, Ly, z(ind))
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: p
    integer , intent(in)  :: ind
    real(rp), intent(out) :: mean
    integer :: i,j,k
    integer :: ierr
    mean = 0.
    !$acc data copy(mean) async(1)
    !$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)  REDUCTION(+:mean)
    do k=1,ind
      do j=1,n(2)
        do i=1,n(1)
          mean = mean + (p(i,j,k)**2)*grid_vol_ratio(k)
        end do
      end do
    end do
    !$acc end data
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean_mod_sq

  subroutine vort_x(pv,pw,indi,indj,indk,dl,dzf,vort)
    use mod_types
    implicit none
    !
    ! compute streamwise vorticity at x(indi),y(indj),z(indk)
    !
    real(rp), intent(in   ), dimension(0:,0:,0:) :: pv,pw
    integer , intent(in   )                      :: indi,indj,indk
    real(rp), intent(in   ), dimension(3 )       :: dl
    real(rp), intent(in   ), dimension(0:)       :: dzf
    real(rp), intent(  out)                      :: vort
    real(rp)  :: dwdy,dvdz
    
    dwdy = pw(indi,indj+1,indk+1) + pw(indi,indj+1,indk  ) + pw(indi,indj-1,indk  ) + pw(indi,indj-1,indk+1)
    dvdz = pv(indi,indj+1,indk+1) + pv(indi,indj  ,indk  ) + pv(indi,indj  ,indk-1) + pv(indi,indj+1,indk-1)
    vort = 0.25*(dwdy/dl(2) - dvdz/dzf(indk))
  end subroutine vort_x

  pure integer function f_sizeof(val) result(isize)
    !
    ! returns storage size of the scalar argument val in bytes
    !
    implicit none
    class(*), intent(in) :: val
    isize = storage_size(val)/8
  end function f_sizeof
  subroutine swap(arr1,arr2)
    use mod_types, only: rp
    implicit none
    real(rp), intent(inout), pointer, contiguous, dimension(:,:,:) :: arr1,arr2
    real(rp),                pointer, contiguous, dimension(:,:,:) :: tmp
    tmp  => arr1
    arr1 => arr2
    arr2 => tmp
  end subroutine swap
#if defined(_OPENACC)
  function device_memory_footprint(n,n_z) result(itotal)
    !
    ! estimate GPU memory footprint, assuming one MPI task <-> one GPU
    !
    use mod_types, only: i8,rp
    integer, intent(in), dimension(3) :: n,n_z
    integer :: nh(3)
    integer(i8) :: itotal,itemp,rp_size
    rp_size = f_sizeof(1._rp)
    itotal = 0
    !
    ! 1. 'main' arrays: u,v,w,p,pp
    !
    nh(:) = 1
    itotal = itotal + product(n(:)+2*nh(:))*rp_size*5
    !
    ! 2. grids arrays: zc,zf,dzc,dzf,dzci,dzfi,grid_vol_ratio_c,grid_vol_ratio_f (tiny footprint)
    !
    nh(:) = 1
    itotal = itotal + (n(3)+2*nh(3))*rp_size*8
    !
    ! 3. solver eigenvalues and Gauss elimination coefficient arrays (small footprint)
    !    rhs?%[x,y,z] arrays, lambdaxy? arrays, and a?,b?,c? arrays
    !
    block
      integer(i8) :: itemp1,itemp1_(3),itemp2,itemp3
      itemp1_(:) = [n_z(2)*n_z(3)*2,n_z(1)*n_z(3)*2,n_z(1)*n_z(2)*2]
      itemp1 = sum(itemp1_(:))   ! rhs
      itemp2 = product(n_z(1:2)) ! lambdaxy
      itemp3 = n_z(3)*3          ! a,b,c
#if   !defined(_IMPDIFF)
      !
      ! rhsbp, lambdaxyp, ap,bp,cp
      !
      itotal = itotal + itemp1*rp_size                + itemp2*rp_size   + itemp3*rp_size
#elif  defined(_IMPDIFF_1D)
      !
      ! rhsbp,rhsb[u,v,w,buf]%z, lambdaxyp, a?,b?,c? [p,u,v,w,buf]
      !
      itotal = itotal + (itemp1+itemp1_(3)*4)*rp_size + itemp2*rp_size   + itemp3*rp_size*5
#else
      !
      ! rhsbp,rhsb[u,v,w,buf]%[x,y,z], lambdaxy[p,u,v,w], (a?,b?,c?)[p,u,v,w,buf]
      !
      itotal = itotal + itemp1*rp_size*(1+4)          + itemp2*rp_size*5 + itemp3*rp_size*5
#endif
    end block
    !
    ! 4. prediction velocity arrays arrays d[u,v,w]dtrk_t, d[u,v,w]dtrko_t
    !
    itemp  = product(n(:))*rp_size
    itotal = itotal + itemp*6
#if defined(_IMPDIFF)
    itotal = itotal + itemp*3
#endif
    !
    ! 5. transpose & FFT buffer arrays, halo buffer arrays, and solver arrays
    !    taken directly from `mod_common_cudecomp`
    !
    block
      use mod_common_cudecomp, only: work,work_halo,solver_buf_0,solver_buf_1,pz_aux_1,pz_aux_2
      itemp = storage_size(work        ,i8)*size(work        ) + storage_size(work_halo   ,i8)*size(work_halo   ) + &
              storage_size(solver_buf_0,i8)*size(solver_buf_0) + storage_size(solver_buf_1,i8)*size(solver_buf_1) + &
              storage_size(pz_aux_1    ,i8)*size(pz_aux_1    ) + storage_size(pz_aux_2    ,i8)*size(pz_aux_2    )
      itotal = itotal + itemp/8
    end block
  end function device_memory_footprint
#endif
end module mod_utils
