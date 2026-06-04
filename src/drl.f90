module mod_drl
  use mod_types
  use mod_common_mpi, only: myid,ierr,intracomm,canscomm,ourid
  use mpi
  implicit none
  public
  real(rp) :: n_act, n_eps, offset
  integer  :: bnx, bny, n_agents, num_files, n_act_interval
  integer , dimension(2) :: bn_arr,n_arr
  real(rp) :: drl_dt, drl_dtn, drl_dpdx
  logical  :: drl_flag
  integer  :: drl_ind_mean, drl_inc, drl_ind_sample
  logical :: send_drl, start_drl
  character(len=5) :: req
  ! new IO for non-continuous episodes
  integer :: prev_istep, ifile_drl
  real(rp) :: prev_time
  character(len=4)   :: file_num_drl
  character(len=256) :: filename_drl
  real(rp) :: rand_drl

  ! SURD multi-z dump (optional, gated by drl_surd_dump in drl.nml)
  integer, parameter :: drl_n_planes_max = 16
  logical :: drl_surd_dump
  integer :: drl_n_planes, drl_n_slice
  integer, dimension(drl_n_planes_max) :: drl_ind_sample_surd
  integer :: drl_surd_step
  real(rp), allocatable :: u_planes(:,:,:), v_planes(:,:,:), w_planes(:,:,:), p_planes(:,:,:)
  real(rp), allocatable :: surd_scalars(:,:)

  real(rp), allocatable   :: u_obs(:,:), w_obs(:,:), p_obs(:,:), t_obs(:,:)
  real(rp), allocatable   :: var_opp_marl(:,:)
  contains
  !
  subroutine get_obs_drl(p,ind,inc,obs_num)
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: p
    integer , intent(in   )  :: ind,inc
    real(rp), intent(in   )  :: obs_num
    !
    integer                  :: i,j,nx,ny
    character(len=10)        :: obs_id
    character(len=10)        :: obs_id_num
    !
    nx = size(p,1)-2
    ny = size(p,2)-2
    !
    write(obs_id    ,'(i0)') myid
    write(obs_id_num,'(i0)') nint(obs_num)
    !
    open(1,file='obs/obs-'//trim(obs_id)//'_'//trim(obs_id_num)//'.dat')
    do i=1,nx,inc
      do j=1,ny,inc
       write(1,'(f7.5)') p(i,j,ind)
      end do
    end do
    close(1)
  end subroutine get_obs_drl

  subroutine get_obs_mpi1d(p,ind,inc,dims,p_obs)
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: p
    integer , intent(in   )  :: ind,inc,dims
    real(rp), intent(  out), dimension(0:)       :: p_obs
    !
    integer                  :: i,j,nx,ny,k
    !
    nx = size(p,1)-2
    ny = size(p,2)-2
    !
    k=0
    do i=1,nx,inc
      do j=1,ny,inc
        p_obs(k) = p(i,j,ind)
        k = k+1
      end do
    end do
    close(1)
  end subroutine get_obs_mpi1d

  subroutine get_obs_mpi(p,ind,inc,p_obs)
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: p
    integer , intent(in   )  :: ind,inc
    real(rp), intent(  out), dimension(0:,0:)    :: p_obs
    !
    !call undersample(p,ind,inc,p_obs)
 
  end subroutine get_obs_mpi


  subroutine get_obs_drl_vort(pv,pw,dl,dzf,ind,inc,obs_num)
    use mod_utils, only: vort_x
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: pv,pw
    real(rp), intent(in   ), dimension(3 )       :: dl
    real(rp), intent(in   ), dimension(0:)       :: dzf
    integer , intent(in   )  :: ind,inc
    real(rp), intent(in   )  :: obs_num
    !
    integer                  :: i,j,nx,ny
    real(rp)                 :: vort
    character(len=10)        :: obs_id
    character(len=10)        :: obs_id_num
    !
    nx = size(pv,1)-2
    ny = size(pv,2)-2
    !
    write(obs_id    ,'(i0)') myid
    write(obs_id_num,'(i0)') nint(obs_num)
    !
    open(1,file='obs/obs-'//trim(obs_id)//'_'//trim(obs_id_num)//'.dat')
    do i=1,nx,inc
      do j=1,ny,inc
       call vort_x(pv,pw,i,j,ind,dl,dzf,vort)
       write(1,*) vort
      end do
    end do
    close(1)
  end subroutine get_obs_drl_vort


  subroutine read_drl(myid)
    use mpi
    implicit none
    integer , intent(in   ) :: myid
    integer :: ierr, id_in
    namelist /opp_drl/ &
                      n_act,   &
                      n_eps,   &
                      bnx,     &
                      bny,     &
                      drl_inc, &
                      n_agents,&
                      num_files,&
                      n_act_interval, &
                      drl_ind_sample, &
                      drl_surd_dump,  &
                      drl_n_planes,   &
                      drl_ind_sample_surd, &
                      drl_n_slice
    n_act_interval = 10
    drl_ind_sample = 20    ! default keeps every existing run dir byte-identical
    drl_surd_dump = .false.
    drl_n_planes = 1
    drl_ind_sample_surd = 0
    drl_ind_sample_surd(1) = 20
    drl_n_slice = 10
    drl_surd_step = 0
    id_in = 101
    open(newunit=id_in,file='drl.nml',status='old',action='read',iostat=ierr)
      if (ierr==0) then
        read(id_in,nml=opp_drl,iostat=ierr)
      else
        if (myid==0) print*, 'Error reading DRL file'
        if (myid==0) print*, 'Aborting...'
        call MPI_FINALIZE(ierr)
        error stop
      end if
    close(id_in)
  end subroutine read_drl
  
  subroutine write_drl_io(myid,t_max,dpdx,amps) !! only working with stw, but not needed
    implicit none
    integer , intent(in) :: myid
    real(rp), intent(in) :: t_max, dpdx
    real(rp), intent(in) :: amps(:)
    integer :: id_out
    
    id_out = 102
    open(newunit=id_out,file='drl.nml',status='replace')
      write(id_out,*) '&stw_drl'
      write(id_out,*) 'n_act=' , n_act+1.
      write(id_out,*) 'offset=', offset + t_max*amps(1)
      !write(id_out,*) ''
    close(id_out)
  
    id_out = 103
    open(newunit=id_out,file='dpdx.dat')
      write(id_out,*) dpdx
    close(id_out)
  end subroutine write_drl_io

  subroutine init_drl_var(id)
    implicit none
    integer , intent(in   ) :: id
    integer :: sync_flag

    ! Wait for Python to be ready
    ! if (id == 0) print*, 'Fortran: Waiting for Python ready signal'
    call MPI_BCAST(sync_flag, 1, MPI_INTEGER, 0, intracomm, ierr)
    ! if (id == 0) print*, 'Fortran: Received Python ready signal'

    drl_flag  = .false.
    drl_dt    = 0.0335
    drl_dtn   = 0.034
    drl_dpdx  = 0.
    drl_ind_mean    = 54 ! corresponding to y+=100
    ! drl_ind_sample is now read from &opp_drl in drl.nml (default 20,
    ! matches the historical hard-coded value: zc(20)=0.0765 -> y+=14.1
    ! on the run0_gru training grid Re_tau=184, gtype=2, gr=1.5).
    ! Backward compat preserved by the default; run0_theory_surd sets it
    ! to 25 in its drl.nml so the detection plane lands at z+~13.43 on
    ! the gr=1.8 evaluation grid (closest match to the training z+).
    u_obs         = 0.
    w_obs         = 0.
    p_obs         = 0.
    send_drl      = .false.
    req           = 'NULLL'
    n_act     = 1
    start_drl = .true.
    send_drl  = .false. 
    n_eps = 1800.
    
    ! if (id == 0) print*, 'DRL vars initialised'
  end subroutine init_drl_var

subroutine drl_read_request(re, dpdx, uobs, wobs, pobs, done, kill)
    implicit none
    character(len=5), intent(in) :: re
    real(rp), intent(inout) :: dpdx
    real(rp), intent(inout), dimension(:,:) :: uobs, wobs, pobs
    logical , intent(inout) :: done, kill

    if (re == 'START' .or. re == 'CONTN') then
        ! Reset values
        dpdx = 0.
        uobs = 0.
        wobs = 0.
        pobs = 0.

        ! Single rank: receive action array directly from Python (rank 0 in intracomm)
        call MPI_RECV(var_opp_marl, size(var_opp_marl), MPI_DOUBLE, 0, 1, intracomm, MPI_STATUS_IGNORE, ierr)
        !$acc update device(var_opp_marl)

    else if (re == 'ENDED') then
        done = .true.
        kill = .true.
    else
        print*, 'Wrong message sent from Python: ', re, ', aborting...'
        done = .true.
        kill = .true.
    end if

end subroutine drl_read_request

  subroutine drl_end_episode(re,done,kill)
    implicit none
    character(len=5), intent(in) :: re
    logical , intent(inout) :: done, kill

    if (re == 'CONTN') then
      ! if (myid == 0) print*, re, ' executed at the end'
      done = .false.
    else if (re == 'CLOSE') then
      ! if (myid == 0) print*, re, ' executed at the end'
      done = .true.
    else if (re == 'CONTR') then
      ! if (myid == 0) print*, re, ' executed at the end'
      done = .false.
    else if (re == 'ENDED') then
      if (myid==0) print*, 'TRAINING DONE, check saved data'
      done = .true.
      kill = .true.
    else   
      if (myid==0) print*, 'Something went wrong with the communication, aborting'
      if (myid==0) print*, 'I received ', re
      done = .true.
      kill = .true.
    end if
  end subroutine drl_end_episode

  subroutine undersample(p,ind,inc,p_us,idd)
  implicit none
  real(rp), intent(in   ), dimension(0:,0:,0:) :: p
  integer , intent(in   ) :: ind,inc
  real(rp), intent(  out), dimension(:,:)    :: p_us
  integer , intent(in   ) :: idd
  integer :: nx,ny,i,j
  
  nx = size(p,1)-2
  ny = size(p,2)-2

  if ((mod(nx,inc).ne.0).or.(mod(ny,inc).ne.0)) then
    error stop "Something went wrong - nx or ny are not multiples of inc"
  end if

  do i = 1,nx,inc
    do j = 1,ny,inc
      if (inc.ne.1) then
        p_us((i+inc-1)/inc,(j+inc-1)/inc) = p(i,j,ind)
        !p_us((i+1)/inc,(j+1)/inc2) = real(idd)
      else
        p_us(i,j) = p(i,j,ind)
      end if
    enddo
  enddo
  
  end subroutine undersample

  subroutine undersample_tau(p,inc,p_us,idd,visc,dzi)
  implicit none
  real(rp), intent(in   ), dimension(0:,0:,0:) :: p
  integer , intent(in   ) :: inc
  real(rp), intent(  out), dimension(:,:)    :: p_us
  integer , intent(in   ) :: idd
  real(rp), intent(in   ) :: visc, dzi
  integer :: nx,ny,i,j
  
  nx = size(p,1)-2
  ny = size(p,2)-2

  if ((mod(nx,inc).ne.0).or.(mod(ny,inc).ne.0)) then
    error stop "Something went wrong - nx or ny are not multiples of inc"
  end if

  do i = 1,nx,inc
    do j = 1,ny,inc
      if (inc.ne.1) then
        p_us((i+inc-1)/inc,(j+inc-1)/inc) = (p(i,j,1) - p(i,j,0))*visc*dzi
      else
        p_us(i,j) = (p(i,j,1) - p(i,j,0))*visc*dzi
      end if
    enddo
  enddo
  
  end subroutine undersample_tau  

  subroutine interp_obs(mat_old,mat_new)
  implicit none
  real(rp), intent(in   ), dimension(:,:) :: mat_old
  real(rp), intent(  out), dimension(:,:) :: mat_new
  integer  :: new_x, new_y, old_x, old_y
  integer  :: i, j
  real(rp) :: ratio, x_mapped, y_mapped
  integer  :: x_low, y_low
  real(rp) :: x_diff, y_diff
  real(rp) :: tl, tr, bl, br


  old_x = size(mat_old,1)
  old_y = size(mat_old,2)
  new_x = size(mat_new,1)
  new_y = size(mat_new,2)

  ratio = old_x/new_x

  do i = 1, new_x
    do j = 1, new_y
      x_mapped = (i - 1) * ratio
      y_mapped = (j - 1) * ratio
      x_low = FLOOR(x_mapped)
      y_low = FLOOR(y_mapped)
      x_diff = x_mapped - x_low
      y_diff = y_mapped - y_low

      if (x_low >= old_x - 1) x_low = old_x - 2
      if (y_low >= old_y - 1) y_low = old_y - 2

      tl = mat_old(x_low + 1, y_low + 1)
      tr = mat_old(x_low + 2, y_low + 1)
      bl = mat_old(x_low + 1, y_low + 2)
      br = mat_old(x_low + 2, y_low + 2)

      mat_new(i, j) = (1.0 - x_diff) * (1.0 - y_diff) * tl + &
                      x_diff * (1.0 - y_diff) * tr + &
                      (1.0 - x_diff) * y_diff * bl + &
                      x_diff * y_diff * br
    end do
  end do
  end subroutine interp_obs

  subroutine surd_alloc(nx, ny)
    implicit none
    integer, intent(in) :: nx, ny
    if (.not. drl_surd_dump) return
    if (drl_n_planes < 1 .or. drl_n_planes > drl_n_planes_max) then
      print*, 'ERROR: drl_n_planes out of [1,', drl_n_planes_max, ']: ', drl_n_planes
      error stop
    end if
    allocate(u_planes(nx, ny, drl_n_planes))
    allocate(v_planes(nx, ny, drl_n_planes))
    allocate(w_planes(nx, ny, drl_n_planes))
    allocate(p_planes(nx, ny, drl_n_planes))
    ! per-plane scalar bundle, 9 numbers per plane:
    ! 1=<u>, 2=<w>, 3=<u^2>, 4=<v^2>, 5=<w^2>, 6=<p^2>, 7=<u*w>, 8=<u*v>, 9=<v*w>
    allocate(surd_scalars(9, drl_n_planes))
    u_planes = 0._rp; v_planes = 0._rp; w_planes = 0._rp; p_planes = 0._rp
    surd_scalars = 0._rp
  end subroutine surd_alloc

  subroutine surd_compute_and_send(u, v, w, p, nx, ny)
    implicit none
    real(rp), intent(in), dimension(0:,0:,0:) :: u, v, w, p
    integer, intent(in) :: nx, ny
    integer :: kp, ind, i, j, tag, field_idx
    real(rp) :: norm, su, sw, suu, svv, sww, spp, suw, suv, svw
    real(rp) :: uu, vv, ww, pp
    logical :: do_slice

    if (.not. drl_surd_dump) return

    drl_surd_step = drl_surd_step + 1
    do_slice = (mod(drl_surd_step, drl_n_slice) == 0)
    norm = 1._rp / real(nx*ny, rp)

    do kp = 1, drl_n_planes
      ind = drl_ind_sample_surd(kp)
      if (ind < 0 .or. ind > size(u,3)-1) then
        print*, 'ERROR: drl_ind_sample_surd(', kp, ')=', ind, ' out of bounds'
        error stop
      end if
      ! patch-averaged scalars at this plane (single-rank CaNS: direct sum)
      su = 0._rp; sw = 0._rp
      suu = 0._rp; svv = 0._rp; sww = 0._rp; spp = 0._rp
      suw = 0._rp; suv = 0._rp; svw = 0._rp
      do j = 1, ny
        do i = 1, nx
          uu = u(i,j,ind); vv = v(i,j,ind); ww = w(i,j,ind); pp = p(i,j,ind)
          su  = su  + uu
          sw  = sw  + ww
          suu = suu + uu*uu
          svv = svv + vv*vv
          sww = sww + ww*ww
          spp = spp + pp*pp
          suw = suw + uu*ww
          suv = suv + uu*vv
          svw = svw + vv*ww
          if (do_slice) then
            u_planes(i,j,kp) = uu
            v_planes(i,j,kp) = vv
            w_planes(i,j,kp) = ww
            p_planes(i,j,kp) = pp
          end if
        end do
      end do
      surd_scalars(1, kp) = su  * norm
      surd_scalars(2, kp) = sw  * norm
      surd_scalars(3, kp) = suu * norm
      surd_scalars(4, kp) = svv * norm
      surd_scalars(5, kp) = sww * norm
      surd_scalars(6, kp) = spp * norm
      surd_scalars(7, kp) = suw * norm
      surd_scalars(8, kp) = suv * norm
      surd_scalars(9, kp) = svw * norm
    end do

    ! send scalar bundle on tag 200 (single contiguous block: 9 * n_planes doubles)
    call MPI_SEND(surd_scalars, size(surd_scalars), MPI_DOUBLE, 0, 200, intracomm, ierr)

    ! send slice block at the slow cadence (4 * n_planes 2D slices)
    if (do_slice) then
      do kp = 1, drl_n_planes
        do field_idx = 0, 3
          tag = 100 + 10*kp + field_idx
          select case (field_idx)
          case (0); call MPI_SEND(u_planes(:,:,kp), nx*ny, MPI_DOUBLE, 0, tag, intracomm, ierr)
          case (1); call MPI_SEND(v_planes(:,:,kp), nx*ny, MPI_DOUBLE, 0, tag, intracomm, ierr)
          case (2); call MPI_SEND(w_planes(:,:,kp), nx*ny, MPI_DOUBLE, 0, tag, intracomm, ierr)
          case (3); call MPI_SEND(p_planes(:,:,kp), nx*ny, MPI_DOUBLE, 0, tag, intracomm, ierr)
          end select
        end do
      end do
    end if
  end subroutine surd_compute_and_send

end module mod_drl
