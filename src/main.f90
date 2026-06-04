! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
!
!        CCCCCCCCCCCCC                    NNNNNNNN        NNNNNNNN    SSSSSSSSSSSSSSS
!     CCC::::::::::::C                    N:::::::N       N::::::N  SS:::::::::::::::S
!   CC:::::::::::::::C                    N::::::::N      N::::::N S:::::SSSSSS::::::S
!  C:::::CCCCCCCC::::C                    N:::::::::N     N::::::N S:::::S     SSSSSSS
! C:::::C       CCCCCC   aaaaaaaaaaaaa    N::::::::::N    N::::::N S:::::S
!C:::::C                 a::::::::::::a   N:::::::::::N   N::::::N S:::::S
!C:::::C                 aaaaaaaaa:::::a  N:::::::N::::N  N::::::N  S::::SSSS
!C:::::C                          a::::a  N::::::N N::::N N::::::N   SS::::::SSSSS
!C:::::C                   aaaaaaa:::::a  N::::::N  N::::N:::::::N     SSS::::::::SS
!C:::::C                 aa::::::::::::a  N::::::N   N:::::::::::N        SSSSSS::::S
!C:::::C                a::::aaaa::::::a  N::::::N    N::::::::::N             S:::::S
! C:::::C       CCCCCC a::::a    a:::::a  N::::::N     N:::::::::N             S:::::S
!  C:::::CCCCCCCC::::C a::::a    a:::::a  N::::::N      N::::::::N SSSSSSS     S:::::S
!   CC:::::::::::::::C a:::::aaaa::::::a  N::::::N       N:::::::N S::::::SSSSSS:::::S
!     CCC::::::::::::C  a::::::::::aa:::a N::::::N        N::::::N S:::::::::::::::SS
!        CCCCCCCCCCCCC   aaaaaaaaaa  aaaa NNNNNNNN         NNNNNNN  SSSSSSSSSSSSSSS
!-------------------------------------------------------------------------------------
! CaNS -- Canonical Navier-Stokes Solver
!-------------------------------------------------------------------------------------
program cans
#if defined(_DEBUG)
  use, intrinsic :: iso_fortran_env, only: compiler_version,compiler_options
#endif
  use, intrinsic :: iso_c_binding  , only: C_PTR
  use, intrinsic :: ieee_arithmetic, only: is_nan => ieee_is_nan
  use mpi
  use decomp_2d
  use mod_bound          , only: boundp,bounduvw,updt_rhs_b
  use mod_chkdiv         , only: chkdiv
  use mod_chkdt          , only: chkdt
  use mod_common_mpi     , only: myid,ourid,ierr,intracomm,canscomm,mysize,oursize
  use mod_correc         , only: correc
  use mod_fft            , only: fftini,fftend
  use mod_fillps         , only: fillps
  use mod_initflow       , only: initflow
  use mod_initgrid       , only: initgrid
  use mod_initmpi        , only: initmpi
  use mod_initsolver     , only: initsolver
  use mod_load           , only: load_all
  use mod_mom            , only: bulk_forcing, cmpt_wallshear
  use mod_rk             , only: rk
  use mod_output         , only: out0d,gen_alias,out1d,out1d_chan,out2d,out3d,write_log_output,write_visu_2d,write_visu_3d
  use mod_param          , only: ng,l,dl,dli, &
                                 gtype,gr, &
                                 cfl,dtmax,dt_f, &
                                 visc, &
                                 inivel,is_wallturb, &
                                 nstep,time_max,tw_max,stop_type, &
                                 restart,is_overwrite_save,nsaves_max, &
                                 icheck,iout0d,iout1d,iout2d,iout3d,isave, &
                                 cbcvel,bcvel,cbcpre,bcpre, &
                                 is_forced,bforce,velf, &
                                 dims, &
                                 nb,is_bound, &
                                 rkcoeff,small, &
                                 datadir,   &
                                 read_input, time
  use mod_sanity         , only: test_sanity_input,test_sanity_solver
#if !defined(_OPENACC)
  use mod_solver         , only: solver
#if defined(_IMPDIFF_1D)
  use mod_solver         , only: solver_gaussel_z
#endif
#else
  use mod_solver_gpu     , only: solver => solver_gpu
#if defined(_IMPDIFF_1D)
  use mod_solver_gpu     , only: solver_gaussel_z => solver_gaussel_z_gpu
#endif
  use mod_workspaces     , only: init_wspace_arrays,set_cufft_wspace
  use mod_common_cudecomp, only: istream_acc_queue_1
#endif
  use mod_timer          , only: timer_tic,timer_toc,timer_print
  use mod_updatep        , only: updatep
  use mod_utils          , only: bulk_mean, bulk_mean_mod, bulk_mean_mod_sq, bulk_mean_2d
  !@acc use mod_utils    , only: device_memory_footprint
  use mod_types
  use mod_opposition      
  use omp_lib
  use mod_drl
  implicit none
  integer , dimension(3) :: lo,hi,n,n_x_fft,n_y_fft,lo_z,hi_z,n_z
  real(rp), allocatable, dimension(:,:,:) :: u,v,w,p,pp
  real(rp), dimension(0:1,3) :: tauxo,tauyo,tauzo
  real(rp), dimension(3) :: f
  real(rp), dimension(0:1,3) :: taux,tauy,tauz
#if !defined(_OPENACC)
  type(C_PTR), dimension(2,2) :: arrplanp
#else
  integer    , dimension(2,2) :: arrplanp
#endif
  real(rp), allocatable, dimension(:,:) :: lambdaxyp
  real(rp), allocatable, dimension(:) :: ap,bp,cp
  real(rp) :: normfftp
  type rhs_bound
    real(rp), allocatable, dimension(:,:,:) :: x
    real(rp), allocatable, dimension(:,:,:) :: y
    real(rp), allocatable, dimension(:,:,:) :: z
  end type rhs_bound
  type(rhs_bound) :: rhsbp
  real(rp) :: alpha
#if defined(_IMPDIFF)
#if !defined(_OPENACC)
  type(C_PTR), dimension(2,2) :: arrplanu,arrplanv,arrplanw
#else
  integer    , dimension(2,2) :: arrplanu,arrplanv,arrplanw
#endif
  real(rp), allocatable, dimension(:,:) :: lambdaxyu,lambdaxyv,lambdaxyw,lambdaxy
  real(rp), allocatable, dimension(:) :: au,av,aw,bu,bv,bw,cu,cv,cw,aa,bb,cc
  real(rp) :: normfftu,normfftv,normfftw
  type(rhs_bound) :: rhsbu,rhsbv,rhsbw
  real(rp), allocatable, dimension(:,:,:) :: rhsbx,rhsby,rhsbz
#endif
  real(rp) :: dt,dti,dt_cfl,dtrk,dtrki,divtot,divmax
  integer :: irk,istep
  real(rp), allocatable, dimension(:) :: dzc  ,dzf  ,zc  ,zf  ,dzci  ,dzfi, &
                                         dzc_g,dzf_g,zc_g,zf_g,dzci_g,dzfi_g, &
                                         grid_vol_ratio_c,grid_vol_ratio_f
  real(rp) :: meanvelu,meanvelv,meanvelw
  real(rp), dimension(3) :: dpdl
  !real(rp), allocatable, dimension(:) :: var
  real(rp), dimension(42) :: var
#if defined(_TIMING)
  real(rp) :: dt12,dt12av,dt12min,dt12max
#endif
  real(rp) :: twi,tw
  integer  :: savecounter
  character(len=7  ) :: fldnum
  character(len=4  ) :: chkptnum
  character(len=100) :: filename
  integer :: k,kk,i,j
  logical :: is_done,kill
  logical :: pfix,file_exists
  real(rp), allocatable, dimension(:,:,:) :: tau_wall
  !
  call MPI_INIT(ierr)
  ! MPMD mode: Python = rank 0, CaNS = rank 1 in MPI_COMM_WORLD
  ! intracomm = DUP of COMM_WORLD (for Python-CaNS data exchange)
  ! canscomm  = SPLIT with color=1 (CaNS-internal operations)
  call MPI_COMM_DUP(MPI_COMM_WORLD, intracomm, ierr)
  call MPI_COMM_RANK(intracomm, ourid, ierr)
  call MPI_COMM_SIZE(intracomm, oursize, ierr)
  call MPI_COMM_SPLIT(MPI_COMM_WORLD, 1, ourid, canscomm, ierr)
  call MPI_COMM_RANK(canscomm, myid, ierr)
  call MPI_COMM_SIZE(canscomm, mysize, ierr)
  
  !
  ! read parameter file
  !
  call read_input(myid)
  if (myid.eq.0) then
    print*, "ng,l,dl,dli", ng,l,dl,dli
    print*, "gtype,gr", gtype,gr 
    print*, "cfl,dtmax,dt_f", cfl,dtmax,dt_f           
    print*, "visc", visc      
    print*, "inivel,is_wallturb", inivel,is_wallturb      
    print*, "nstep,time_max,tw_max,stop_type", nstep,time_max,tw_max,stop_type      
    print*, "restart,is_overwrite_save,nsaves_max", restart,is_overwrite_save,nsaves_max       
    print*, "icheck,iout0d,iout1d,iout2d,iout3d,isave", icheck,iout0d,iout1d,iout2d,iout3d,isave       
    print*, "cbcvel,bcvel,cbcpre,bcpre", cbcvel,bcvel,cbcpre,bcpre       
    print*, "is_forced,bforce,velf", is_forced,bforce,velf       
    print*, "dims", dims       
  end if          
  ! read opposition control file
  !
      ! Check if file exists
    inquire(file='opp.nml', exist=file_exists)
    if (myid == 0) then
        if (.not. file_exists) then
            print *, "Error: opp.nml file not found!"
            call MPI_ABORT(canscomm, 1, ierr)
        end if
    end if
    
  call read_opposition(myid)
  pfix = .true.

       ! Check if file exists
  inquire(file='drl.nml', exist=file_exists)
  if (myid == 0) then
      if (.not. file_exists) then
          print *, "Error: drl.nml file not found!"
          call MPI_ABORT(canscomm, 1, ierr)
      end if
  end if

  call read_drl(myid)
  if (myid==0) print*, 'DRL settings read: alpha_opp = ', alpha_opp, 'n_walls =', n_walls
  if (myid.eq.0) then
    print*, "n_act = ", n_act, "n_eps = ", n_eps, "bnx =  ", bnx, "bny = ", bny
    print*, "drl_inc = ", drl_inc, "n_agents = ", n_agents, "num_files = ", num_files
  end if
  !
  ! initialize MPI/OpenMP
  !
  !$ call omp_set_num_threads(omp_get_max_threads())
  call initmpi(ng,dims,cbcpre,lo,hi,n,n_x_fft,n_y_fft,lo_z,hi_z,n_z,nb,is_bound)
  twi = MPI_WTIME()
  savecounter = 0
  !
  ! allocate variables
  !
  allocate(u( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           v( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           w( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           p( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           pp(0:n(1)+1,0:n(2)+1,0:n(3)+1), &
           tau_wall(0:n(1)+1,0:n(2)+1,0:n(3)+1))
  allocate(lambdaxyp(n_z(1),n_z(2)))
  allocate(ap(n_z(3)),bp(n_z(3)),cp(n_z(3)))
  allocate(dzc( 0:n(3)+1), &
           dzf( 0:n(3)+1), &
           zc(  0:n(3)+1), &
           zf(  0:n(3)+1), &
           dzci(0:n(3)+1), &
           dzfi(0:n(3)+1))
  allocate(dzc_g( 0:ng(3)+1), &
           dzf_g( 0:ng(3)+1), &
           zc_g(  0:ng(3)+1), &
           zf_g(  0:ng(3)+1), &
           dzci_g(0:ng(3)+1), &
           dzfi_g(0:ng(3)+1))
  allocate(grid_vol_ratio_c,mold=dzc)
  allocate(grid_vol_ratio_f,mold=dzf)
  allocate(rhsbp%x(n(2),n(3),0:1), &
           rhsbp%y(n(1),n(3),0:1), &
           rhsbp%z(n(1),n(2),0:1))
#if defined(_IMPDIFF)
  allocate(lambdaxyu(n_z(1),n_z(2)), &
           lambdaxyv(n_z(1),n_z(2)), &
           lambdaxyw(n_z(1),n_z(2)), &
           lambdaxy( n_z(1),n_z(2)))
  allocate(au(n_z(3)),bu(n_z(3)),cu(n_z(3)), &
           av(n_z(3)),bv(n_z(3)),cv(n_z(3)), &
           aw(n_z(3)),bw(n_z(3)),cw(n_z(3)), &
           aa(n_z(3)),bb(n_z(3)),cc(n_z(3)))
  allocate(rhsbu%x(n(2),n(3),0:1), &
           rhsbu%y(n(1),n(3),0:1), &
           rhsbu%z(n(1),n(2),0:1), &
           rhsbv%x(n(2),n(3),0:1), &
           rhsbv%y(n(1),n(3),0:1), &
           rhsbv%z(n(1),n(2),0:1), &
           rhsbw%x(n(2),n(3),0:1), &
           rhsbw%y(n(1),n(3),0:1), &
           rhsbw%z(n(1),n(2),0:1), &
           rhsbx(  n(2),n(3),0:1), &
           rhsby(  n(1),n(3),0:1), &
           rhsbz(  n(1),n(2),0:1))
#endif
    ! For undersampling, remember that the US ratio must be = inc
  allocate  (u_obs       (1:n(1)/drl_inc,1:n(2)/drl_inc))
  allocate  (w_obs       (1:n(1)/drl_inc,1:n(2)/drl_inc))
  allocate  (p_obs       (1:n(1)/drl_inc,1:n(2)/drl_inc))
  allocate  (t_obs       (1:n(1)/drl_inc,1:n(2)/drl_inc))
  allocate  (var_opp_marl(1:n(1)/drl_inc,1:n(2)/drl_inc))
  var_opp_marl = 0.
  !$acc enter data copyin(var_opp_marl)
  ! SURD multi-z dump arrays (no-op if drl_surd_dump=.false.)
  call surd_alloc(n(1), n(2))
#if defined(_DEBUG)
  if(myid == 0) print*, 'This executable of CaNS was built with compiler: ', compiler_version()
  if(myid == 0) print*, 'Using the options: ', compiler_options()
  block
    character(len=MPI_MAX_LIBRARY_VERSION_STRING) :: mpi_version
    integer :: ilen
    call MPI_GET_LIBRARY_VERSION(mpi_version,ilen,ierr)
    if(myid == 0) print*, 'MPI Version: ', trim(mpi_version)
  end block
  if(myid == 0) print*, ''
#endif
  if(myid == 0) print*, '*******************************'
  if(myid == 0) print*, '*** Beginning of simulation ***'
  if(myid == 0) print*, '*******************************'
  if(myid == 0) print*, ''
  call initgrid(gtype,ng(3),gr,l(3),dzc_g,dzf_g,zc_g,zf_g)
  if(myid == 0) then
    open(99,file=trim(datadir)//'grid.bin',action='write',form='unformatted',access='stream',status='replace')
    write(99) dzc_g(1:ng(3)),dzf_g(1:ng(3)),zc_g(1:ng(3)),zf_g(1:ng(3))
    close(99)
    open(99,file=trim(datadir)//'grid.out')
    do kk=0,ng(3)+1
      write(99,*) 0.,zf_g(kk),zc_g(kk),dzf_g(kk),dzc_g(kk)
    end do
    close(99)
    open(99,file=trim(datadir)//'geometry.out')
      write(99,*) ng(1),ng(2),ng(3)
      write(99,*) l(1),l(2),l(3)
    close(99)
  end if
  print*, 'myid = ', myid, ' hi and lo = ', hi, lo, 'n(1),n(2) = ', n(1),n(2) 
  !$acc enter data copyin(lo,hi,n) async
  !$acc enter data copyin(bforce,dl,dli,l) async
  !$acc enter data copyin(zc_g,zf_g,dzc_g,dzf_g) async
  !$acc enter data create(zc,zf,dzc,dzf,dzci,dzfi,dzci_g,dzfi_g) async
  !
  !$acc parallel loop default(present) private(k) async
  do kk=lo(3)-1,hi(3)+1
    k = kk-(lo(3)-1)
    zc( k) = zc_g(kk)
    zf( k) = zf_g(kk)
    dzc(k) = dzc_g(kk)
    dzf(k) = dzf_g(kk)
    dzci(k) = dzc(k)**(-1)
    dzfi(k) = dzf(k)**(-1)
  end do
  !$acc kernels default(present) async
  dzci_g(:) = dzc_g(:)**(-1)
  dzfi_g(:) = dzf_g(:)**(-1)
  !$acc end kernels
  !$acc enter data create(grid_vol_ratio_c,grid_vol_ratio_f) async
  !$acc kernels default(present) async
  grid_vol_ratio_c(:) = dl(1)*dl(2)*dzc(:)/(l(1)*l(2)*l(3))
  grid_vol_ratio_f(:) = dl(1)*dl(2)*dzf(:)/(l(1)*l(2)*l(3))
  !$acc end kernels
  !$acc update self(zc,zf,dzc,dzf,dzci,dzfi) async
  !$acc exit data copyout(zc_g,zf_g,dzc_g,dzf_g,dzci_g,dzfi_g) async ! not needed on the device
  !$acc wait
  !
  ! test input files before proceeding with the calculation
  !
  call test_sanity_input(ng,dims,stop_type,cbcvel,cbcpre,bcvel,bcpre,is_forced)
  !
  ! initialize Poisson solver
  !
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcpre,bcpre(:,:), &
                  lambdaxyp,['c','c','c'],ap,bp,cp,arrplanp,normfftp,rhsbp%x,rhsbp%y,rhsbp%z)
  !$acc enter data copyin(lambdaxyp,ap,bp,cp) async
  !$acc enter data copyin(rhsbp,rhsbp%x,rhsbp%y,rhsbp%z) async
  !$acc wait
#if defined(_IMPDIFF)
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcvel(:,:,1),bcvel(:,:,1), &
                  lambdaxyu,['f','c','c'],au,bu,cu,arrplanu,normfftu,rhsbu%x,rhsbu%y,rhsbu%z)
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcvel(:,:,2),bcvel(:,:,2), &
                  lambdaxyv,['c','f','c'],av,bv,cv,arrplanv,normfftv,rhsbv%x,rhsbv%y,rhsbv%z)
  call initsolver(ng,n_x_fft,n_y_fft,lo_z,hi_z,dli,dzci_g,dzfi_g,cbcvel(:,:,3),bcvel(:,:,3), &
                  lambdaxyw,['c','c','f'],aw,bw,cw,arrplanw,normfftw,rhsbw%x,rhsbw%y,rhsbw%z)
#if defined(_IMPDIFF_1D)
  deallocate(lambdaxyu,lambdaxyv,lambdaxyw,lambdaxy)
  call fftend(arrplanu)
  call fftend(arrplanv)
  call fftend(arrplanw)
  deallocate(rhsbu%x,rhsbu%y,rhsbv%x,rhsbv%y,rhsbw%x,rhsbw%y,rhsbx,rhsby)
#endif
  !$acc enter data copyin(lambdaxyu,au,bu,cu,lambdaxyv,av,bv,cv,lambdaxyw,aw,bw,cw) async
  !$acc enter data copyin(rhsbu,rhsbu%x,rhsbu%y,rhsbu%z) async
  !$acc enter data copyin(rhsbv,rhsbv%x,rhsbv%y,rhsbv%z) async
  !$acc enter data copyin(rhsbw,rhsbw%x,rhsbw%y,rhsbw%z) async
  !$acc enter data create(lambdaxy,aa,bb,cc) async
  !$acc enter data create(rhsbx,rhsby,rhsbz) async
  !$acc wait
#endif
#if defined(_OPENACC)
  !
  ! determine workspace sizes and allocate the memory
  !
  call init_wspace_arrays()
  call set_cufft_wspace(pack(arrplanp,.true.),istream_acc_queue_1)
#if defined(_IMPDIFF) && !defined(_IMPDIFF_1D)
  call set_cufft_wspace(pack(arrplanu,.true.),istream_acc_queue_1)
  call set_cufft_wspace(pack(arrplanv,.true.),istream_acc_queue_1)
  call set_cufft_wspace(pack(arrplanw,.true.),istream_acc_queue_1)
#endif
  if(myid == 0) print*,'*** Device memory footprint (Gb): ', &
                  device_memory_footprint(n,n_z)/(1._sp*1024**3), ' ***'
#endif
#if defined(_DEBUG_SOLVER)
  call test_sanity_solver(ng,lo,hi,n,n_x_fft,n_y_fft,lo_z,hi_z,n_z,dli,dzc,dzf,dzci,dzfi,dzci_g,dzfi_g, &
                          nb,is_bound,cbcvel,cbcpre,bcvel,bcpre)
#endif
  !
  if(.not.restart) then
    istep = 0
    time = 0.
    call initflow(inivel,bcvel,ng,lo,l,dl,zc,zf,dzc,dzf,visc, &
                  is_forced,velf,bforce,is_wallturb,u,v,w,p)
    if(myid == 0) print*, '*** Initial condition succesfully set ***'
  else
    call load_all('r',trim(datadir)//'fld.bin',canscomm,ng,[1,1,1],lo,hi,u,v,w,p,time,istep)
    istep = 0
    time = 0.
    if(myid == 0) print*, '*** Checkpoint loaded at time = ', time, 'time step = ', istep, '. ***'
  end if
  !$acc enter data copyin(u,v,w,p) create(pp)
  if ((cbcvel(1,3,1) == 'N').and.(cbcvel(1,3,2) == 'N')) then
    allocate(var_opp(1:n(1),1:n(2),0:0))
  else if ((cbcvel(1,3,1) == 'D').and.(cbcvel(1,3,2) == 'D')) then
    allocate(var_opp(1:n(1),1:n(2),0:1))
  end if
  var_opp = 0.
  !$acc enter data copyin(var_opp)
  if (myid == 0) print*, 'eks indexes initialised BEFORE: ', drl_ind_mean, drl_ind_sample
  if (myid == 0) print*, 'Request initialised BEFORE: ', req
  call init_drl_var(myid)
  if (myid == 0) print*, 'eks indexes initialised: ', drl_ind_mean, drl_ind_sample
  if (myid == 0) print*, 'Request initialised: ', req
  ! the first update_opp is with omega given by stw.nml
  call update_opp(n, n_walls, w, var_opp, myid)
  !
  call bounduvw(cbcvel,n,bcvel,nb,is_bound,.false.,dl,dzc,dzf,u,v,w)
  call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,p)
  !
  ! post-process and write initial condition
  !
  write(fldnum,'(i7.7)') istep
  !$acc wait ! not needed but to prevent possible future issues
  !$acc update self(u,v,w,p)
  if(iout1d > 0.and.mod(istep,max(iout1d,1)) == 0) then
    include 'out1d.h90'
  end if
  if(iout2d > 0.and.mod(istep,max(iout2d,1)) == 0) then
    include 'out2d.h90'
  end if
  if(iout3d > 0.and.mod(istep,max(iout3d,1)) == 0) then
    include 'out3d.h90'
  end if
  !
  call chkdt(n,dl,dzci,dzfi,visc,u,v,w,dt_cfl)
  dt = merge(dt_f,min(cfl*dt_cfl,dtmax),dt_f > 0.)
  if(myid == 0) print*, 'dt_cfl = ', dt_cfl, 'dt = ', dt
  dti = 1./dt
  kill = .false.
  !
  ! main loop
  !
  if(myid == 0) print*, '*** Calculation loop starts now ***'
  is_done = .false.
  start_drl = .true.
  do while(.not.is_done)
    !
    ! Receive request from Python
    if (start_drl) then
      call MPI_BCAST(req,5,MPI_CHAR,0,intracomm,ierr)
      call drl_read_request(req,drl_dpdx,u_obs,w_obs,p_obs,is_done,kill)
      start_drl = .false.
    end if
    !
    !
#if defined(_TIMING)
    !$acc wait(1)
    dt12 = MPI_WTIME()
#endif
    istep = istep + 1
    time = time + dt
    if(myid == 0 .and. mod(istep, 20) == 0) print*, 'Time step #', istep, 'Time = ', time
    tauxo(:,:) = 0.; tauyo(:,:) = 0.; tauzo(:,:) = 0.
    dpdl(:)  = 0.
    !
    call update_opp(n, n_walls, w, var_opp, myid)
    call opposition(n,alpha_opp,0,var_opp,w)
    if (n_walls == 2) then
      call opposition(n,alpha_opp,1,var_opp,w)
    end if
    !
    do irk=1,3
      dtrk = sum(rkcoeff(:,irk))*dt
      dtrki = dtrk**(-1)
      call rk(rkcoeff(:,irk),n,dli,dzci,dzfi,grid_vol_ratio_c,grid_vol_ratio_f,visc,dt,p, &
              is_forced,velf,bforce,u,v,w,f)
      call bulk_forcing(n,is_forced,f,u,v,w)
#if defined(_IMPDIFF)
      alpha = -.5*visc*dtrk
      !$OMP PARALLEL WORKSHARE
      !$acc kernels present(rhsbx,rhsby,rhsbz,rhsbu) async(1)
#if !defined(_IMPDIFF_1D)
      rhsbx(:,:,0:1) = rhsbu%x(:,:,0:1)*alpha
      rhsby(:,:,0:1) = rhsbu%y(:,:,0:1)*alpha
#endif
      rhsbz(:,:,0:1) = rhsbu%z(:,:,0:1)*alpha
      !$acc end kernels
      !$OMP END PARALLEL WORKSHARE
      call updt_rhs_b(['f','c','c'],cbcvel(:,:,1),n,is_bound,rhsbx,rhsby,rhsbz,u)
      !$acc kernels default(present) async(1)
      !$OMP PARALLEL WORKSHARE
      aa(:) = au(:)*alpha
      bb(:) = bu(:)*alpha + 1.
      cc(:) = cu(:)*alpha
#if !defined(_IMPDIFF_1D)
      lambdaxy(:,:) = lambdaxyu(:,:)*alpha
#endif
      !$OMP END PARALLEL WORKSHARE
      !$acc end kernels
#if !defined(_IMPDIFF_1D)
      call solver(n,ng,arrplanu,normfftu,lambdaxy,aa,bb,cc,cbcvel(:,:,1),['f','c','c'],u)
#else
      call solver_gaussel_z(n                    ,aa,bb,cc,cbcvel(:,3,1),['f','c','c'],u)
#endif
      !$OMP PARALLEL WORKSHARE
      !$acc kernels present(rhsbx,rhsby,rhsbz,rhsbv) async(1)
#if !defined(_IMPDIFF_1D)
      rhsbx(:,:,0:1) = rhsbv%x(:,:,0:1)*alpha
      rhsby(:,:,0:1) = rhsbv%y(:,:,0:1)*alpha
#endif
      rhsbz(:,:,0:1) = rhsbv%z(:,:,0:1)*alpha
      !$acc end kernels
      !$OMP END PARALLEL WORKSHARE
      call updt_rhs_b(['c','f','c'],cbcvel(:,:,2),n,is_bound,rhsbx,rhsby,rhsbz,v)
      !$acc kernels default(present) async(1)
      !$OMP PARALLEL WORKSHARE
      aa(:) = av(:)*alpha
      bb(:) = bv(:)*alpha + 1.
      cc(:) = cv(:)*alpha
#if !defined(_IMPDIFF_1D)
      lambdaxy(:,:) = lambdaxyv(:,:)*alpha
#endif
      !$OMP END PARALLEL WORKSHARE
      !$acc end kernels
#if !defined(_IMPDIFF_1D)
      call solver(n,ng,arrplanv,normfftv,lambdaxy,aa,bb,cc,cbcvel(:,:,2),['c','f','c'],v)
#else
      call solver_gaussel_z(n                    ,aa,bb,cc,cbcvel(:,3,2),['c','f','c'],v)
#endif
      !$OMP PARALLEL WORKSHARE
      !$acc kernels present(rhsbx,rhsby,rhsbz,rhsbw) async(1)
#if !defined(_IMPDIFF_1D)
      rhsbx(:,:,0:1) = rhsbw%x(:,:,0:1)*alpha
      rhsby(:,:,0:1) = rhsbw%y(:,:,0:1)*alpha
#endif
      rhsbz(:,:,0:1) = rhsbw%z(:,:,0:1)*alpha
      !$acc end kernels
      !$OMP END PARALLEL WORKSHARE
      call updt_rhs_b(['c','c','f'],cbcvel(:,:,3),n,is_bound,rhsbx,rhsby,rhsbz,w)
      !$acc kernels default(present) async(1)
      !$OMP PARALLEL WORKSHARE
      aa(:) = aw(:)*alpha
      bb(:) = bw(:)*alpha + 1.
      cc(:) = cw(:)*alpha
#if !defined(_IMPDIFF_1D)
      lambdaxy(:,:) = lambdaxyw(:,:)*alpha
#endif
      !$OMP END PARALLEL WORKSHARE
      !$acc end kernels
#if !defined(_IMPDIFF_1D)
      call solver(n,ng,arrplanw,normfftw,lambdaxy,aa,bb,cc,cbcvel(:,:,3),['c','c','f'],w)
#else
      call solver_gaussel_z(n                    ,aa,bb,cc,cbcvel(:,3,3),['c','c','f'],w)
#endif
#endif
      dpdl(:) = dpdl(:) + f(:)
      ! var_opp is frozen across the 3 RK sub-steps: it was set once by
      ! update_opp at the top of the timestep, so opposition() inside
      ! bounduvw keeps writing the same wall value at every stage.
      call bounduvw(cbcvel,n,bcvel,nb,is_bound,.false.,dl,dzc,dzf,u,v,w,pfix)
      call fillps(n,dli,dzfi,dtrki,u,v,w,pp)
      call updt_rhs_b(['c','c','c'],cbcpre,n,is_bound,rhsbp%x,rhsbp%y,rhsbp%z,pp)
      if ((pfix).and.(myid==0)) then
        call solver(n,ng,arrplanp,normfftp,lambdaxyp,ap,bp,cp,cbcpre,['c','c','c'],pp,pfix)
      else
        call solver(n,ng,arrplanp,normfftp,lambdaxyp,ap,bp,cp,cbcpre,['c','c','c'],pp)
      end if
      call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,pp)
      call correc(n,dli,dzci,dtrk,pp,u,v,w)
      call bounduvw(cbcvel,n,bcvel,nb,is_bound,.true.,dl,dzc,dzf,u,v,w)
      call updatep(n,dli,dzci,dzfi,alpha,pp,p)
      call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,p)
    end do
    dpdl(:) = -dpdl(:)*dti
    !
    ! check simulation stopping criteria
    !
    if(stop_type(1)) then ! maximum number of time steps reached
      if(istep >= nstep   ) is_done = is_done.or..true.
    end if
    if(stop_type(2)) then ! maximum simulation time reached
      if(time  >= time_max*n_act*n_eps) is_done = is_done.or..true.
    end if
    if(stop_type(3)) then ! maximum wall-clock time reached
      tw = (MPI_WTIME()-twi)/3600.
      if(tw    >= tw_max  ) is_done = is_done.or..true.
    end if
    if(icheck > 0.and.mod(istep,max(icheck,1)) == 0) then
      if(myid == 0) print*, 'Checking stability and divergence...'
      call chkdt(n,dl,dzci,dzfi,visc,u,v,w,dt_cfl)
      dt = merge(dt_f,min(cfl*dt_cfl,dtmax),dt_f > 0.)
      if(myid == 0) print*, 'dt_cfl = ', dt_cfl, 'dt = ', dt
      if(dt_cfl < small) then
        if(myid == 0) print*, 'ERROR: time step is too small.'
        if(myid == 0) print*, 'Aborting...'
        is_done = .true.
        kill = .true.
      end if
      dti = 1./dt
      call chkdiv(lo,hi,dli,dzfi,u,v,w,divtot,divmax)
      if(myid == 0) print*, 'Total divergence = ', divtot, '| Maximum divergence = ', divmax
#if !defined(_MASK_DIVERGENCE_CHECK)
      if(divmax > (small*100000.0).or.is_nan(divtot)) then
        if(myid == 0) print*, 'ERROR: maximum divergence is too large.'
        if(myid == 0) print*, 'Aborting...'
        is_done = .true.
        kill = .true.
      end if
#endif
    end if
    !
    ! output routines below
    !
    if(iout0d > 0.and.mod(istep,max(iout0d,1)) == 0) then
      !allocate(var(4))
      var(1) = 1.*istep
      var(2) = dt
      var(3) = time
      call out0d(trim(datadir)//'time.out',3,var)
      !
      if(any(is_forced(:)).or.any(abs(bforce(:)) > 0.)) then
        meanvelu = 0.
        meanvelv = 0.
        meanvelw = 0.
        if(is_forced(1).or.abs(bforce(1)) > 0.) then
          call bulk_mean(n,grid_vol_ratio_f,u,meanvelu)
        end if
        if(is_forced(2).or.abs(bforce(2)) > 0.) then
          call bulk_mean(n,grid_vol_ratio_f,v,meanvelv)
        end if
        if(is_forced(3).or.abs(bforce(3)) > 0.) then
          call bulk_mean(n,grid_vol_ratio_c,w,meanvelw)
        end if
        if(.not.any(is_forced(:))) dpdl(:) = -bforce(:) ! constant pressure gradient
        var(1)   = time
        var(2:4) = dpdl(1:3)
        var(5:7) = [meanvelu,meanvelv,meanvelw]
        call out0d(trim(datadir)//'forcing.out',7,var)
        call cmpt_wallshear(n,is_forced,is_bound,l,dli,dzci,dzfi,visc,u,v,w,taux,tauy,tauz)
        if (myid.eq.0) print*, "tau_wall = ", taux(1,3)
        call out0d(trim(datadir)//'tau.out',1,[taux(1,3)])
        tau_wall(:,:,1) = (u(:,:,1)-u(:,:,0))*dzci(0)*visc
      end if
    end if
    write(fldnum,'(i7.7)') istep
    if(iout1d > 0.and.mod(istep,max(iout1d,1)) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out1d.h90'
    end if
    if(iout2d > 0.and.mod(istep,max(iout2d,1)) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out2d.h90'
    end if
    if(iout3d > 0.and.mod(istep,max(iout3d,1)) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out3d.h90'
    end if
    if(isave > 0.and.((mod(istep,max(isave,1)) == 0).or.(is_done.and..not.kill))) then
      if(is_overwrite_save) then
        filename = 'fld.bin'
      else
        filename = 'fld_'//fldnum//'.bin'
        if(nsaves_max > 0) then
          if(savecounter >= nsaves_max) savecounter = 0
          savecounter = savecounter + 1
          write(chkptnum,'(i4.4)') savecounter
          filename = 'fld_'//chkptnum//'.bin'
          var(1) = 1.*istep
          var(2) = time
          var(3) = 1.*savecounter
          call out0d(trim(datadir)//'log_checkpoints.out',3,var)
        end if
      end if
      !$acc wait
      !$acc update self(u,v,w,p)
      call load_all('w',trim(datadir)//trim(filename),canscomm,ng,[1,1,1],lo,hi,u,v,w,p,time,istep)
      if(.not.is_overwrite_save) then
        !
        ! fld.bin -> last checkpoint file (symbolic link)
        !
        call gen_alias(myid,trim(datadir),trim(filename),'fld.bin')
      end if
      if(myid == 0) print*, '*** Checkpoint saved at time = ', time, 'time step = ', istep, '. ***'
    end if
#if defined(_TIMING)
      !$acc wait(1)
      dt12 = MPI_WTIME()-dt12
      call MPI_ALLREDUCE(dt12,dt12av ,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
      call MPI_ALLREDUCE(dt12,dt12min,1,MPI_REAL_RP,MPI_MIN,canscomm,ierr)
      call MPI_ALLREDUCE(dt12,dt12max,1,MPI_REAL_RP,MPI_MAX,canscomm,ierr)
      if(myid == 0 .and. mod(istep, 20) == 0) then
        print*, 'Avrg, min & max elapsed time: '
        print*, dt12av/(1.*product(dims)),dt12min,dt12max
      end if
#endif
    !
    ! Here variables useful to eks, dpdx etc are computed at each step for DRL
    drl_dpdx = drl_dpdx + dpdl(1)/real(n_act_interval,rp)
    !
    if (mod(istep,n_act_interval)==0) then
      n_act     = n_act + 1
      is_done   = .false.
      send_drl  = .true.
      start_drl = .true.
    end if
    
    if (send_drl) then
      !$acc wait
      !$acc update self(u,w,p)
      call undersample(u,drl_ind_sample,drl_inc,u_obs,ourid)
      call undersample(w,drl_ind_sample,drl_inc,w_obs,ourid)
      call undersample(p,1,drl_inc,p_obs,ourid) ! wall-adjacent cell (z_c(1)) for wall-work diagnostic
      call undersample(tau_wall,1,1,t_obs,ourid)
      !
      ! Single rank: send observations directly to Python (rank 0 in intracomm)
      call MPI_SEND(u_obs,   size(u_obs),   MPI_DOUBLE,0,5,intracomm,ierr)
      call MPI_SEND(w_obs,   size(w_obs),   MPI_DOUBLE,0,9,intracomm,ierr)
      call MPI_SEND(p_obs,   size(p_obs),   MPI_DOUBLE,0,8,intracomm,ierr)
      call MPI_SEND(t_obs,   size(t_obs),   MPI_DOUBLE,0,7,intracomm,ierr)
      call MPI_SEND(drl_dpdx,1,             MPI_DOUBLE,0,4,intracomm,ierr)
      ! SURD multi-z dump (no-op when drl_surd_dump=.false.); needs an
      ! up-to-date copy of v too because vorticity is post-processed
      ! Python-side from (u,v,w) at each plane.
      if (drl_surd_dump) then
        !$acc update self(v)
        call surd_compute_and_send(u, v, w, p, n(1), n(2))
      end if
      send_drl = .false.
      call MPI_BCAST(req,5,MPI_CHAR,0,intracomm,ierr)
      call drl_end_episode(req,is_done,kill)
      ! Load new field for non-continuous learning
      if (req == 'CONTR') then
        call MPI_BARRIER(canscomm, ierr)  ! Ensure all processes are here
        prev_time = time
        prev_istep = istep

        call MPI_BARRIER(canscomm, ierr)  ! Ensure all processes are synchronized
        call load_all('r',trim(datadir)//'fld.bin',canscomm,ng,[1,1,1],lo,hi,u,v,w,p,time,istep)

        time = prev_time
        istep = prev_istep
        if(myid == 0) then
            print*, '*** Loaded fld.bin for new episode ***'
        endif
        is_done = .false.
        !$acc update device(u,v,w,p)
        call bounduvw(cbcvel,n,bcvel,nb,is_bound,.false.,dl,dzc,dzf,u,v,w)
        call boundp(cbcpre,n,bcpre,nb,is_bound,dl,dzc,p)
      end if
      if (req=='ENDED') is_done = .true.
    end if

    ! Check array bounds before operations
    if (drl_ind_sample > size(u,3)) then
        print*, 'ERROR: drl_ind_sample out of bounds'
        call MPI_Abort(canscomm, 1, ierr)
    endif
  end do
  !
  ! clear ffts
  !
  call fftend(arrplanp)
#if defined(_IMPDIFF) && !defined(_IMPDIFF_1D)
  call fftend(arrplanu)
  call fftend(arrplanv)
  call fftend(arrplanw)
#endif
  if(myid == 0.and.(.not.kill)) print*, '*** Fim ***'
  call decomp_2d_finalize
  call MPI_COMM_FREE(canscomm, ierr)
  call MPI_COMM_FREE(intracomm, ierr)
  call MPI_FINALIZE(ierr)
end program cans
