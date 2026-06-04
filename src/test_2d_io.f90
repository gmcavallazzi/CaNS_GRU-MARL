! -
!
! Test module for 2D slice read/write verification
!
! -
module mod_test_2d_io
  use mpi
  use decomp_2d
  use mod_common_mpi, only: myid, ierr, canscomm
  use mod_output, only: out2d
  use mod_types
  implicit none
  private
  public :: test_2d_slice_io

contains

  subroutine test_2d_slice_io(datadir, ipencil, n, ng, w)
    !
    ! Test that we can correctly read back a 2D slice written with out2d
    ! Writes a slice, reads it back, and writes it again for comparison
    !
    implicit none
    character(len=*), intent(in) :: datadir
    integer, intent(in) :: ipencil
    integer, dimension(3), intent(in) :: n, ng
    real(rp), dimension(1:,1:,1:), intent(in) :: w
    real(rp), allocatable, dimension(:,:) :: plane_read
    integer :: islice, n1, n2
    integer :: fh
    integer(kind=MPI_OFFSET_KIND) :: disp
    integer, dimension(2) :: sizes, subsizes, starts
    integer :: type_plane
    character(len=256) :: fname_orig, fname_test

    if(myid == 0) print*, '*** [TEST] Starting 2D slice read/write test ***'

    ! Test parameters
    islice = ng(3)/2  ! Middle slice in z-direction
    fname_orig = trim(datadir)//'test_w_original.bin'
    fname_test = trim(datadir)//'test_w_readback.bin'

    ! Step 1: Write the plane using existing out2d
    call out2d(fname_orig, 3, islice, w)
    if(myid == 0) print*, '*** [TEST] Original slice written to: ', trim(fname_orig)

    ! Step 2: Determine local array sizes based on pencil decomposition
    ! For XY plane (iplane=3), we need nx_global x ny_global data
    select case(ipencil)
    case(1) ! X-pencil
      n1 = xsize(1)
      n2 = xsize(2)
      sizes = [nx_global, ny_global]
      subsizes = [xsize(1), xsize(2)]
      starts = [xstart(1)-1, xstart(2)-1]
    case(2) ! Y-pencil
      n1 = ysize(1)
      n2 = ysize(2)
      sizes = [nx_global, ny_global]
      subsizes = [ysize(1), ysize(2)]
      starts = [ystart(1)-1, ystart(2)-1]
    case(3) ! Z-pencil
      n1 = zsize(1)
      n2 = zsize(2)
      sizes = [nx_global, ny_global]
      subsizes = [zsize(1), zsize(2)]
      starts = [zstart(1)-1, zstart(2)-1]
    end select

    allocate(plane_read(n1,n2))

    ! Step 3: Read the plane back using MPI-IO
    call MPI_FILE_OPEN(canscomm, fname_orig, MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
    call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
         MPI_ORDER_FORTRAN, MPI_REAL_RP, type_plane, ierr)
    call MPI_TYPE_COMMIT(type_plane, ierr)
    disp = 0_MPI_OFFSET_KIND
    call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL_RP, type_plane, 'native', MPI_INFO_NULL, ierr)
    call MPI_FILE_READ_ALL(fh, plane_read, product(subsizes), MPI_REAL_RP, MPI_STATUS_IGNORE, ierr)
    call MPI_TYPE_FREE(type_plane, ierr)
    call MPI_FILE_CLOSE(fh, ierr)
    if(myid == 0) print*, '*** [TEST] Slice read back successfully'

    ! Step 4: Write the read-back data to a new file
    call MPI_FILE_OPEN(canscomm, fname_test, MPI_MODE_CREATE+MPI_MODE_WRONLY, MPI_INFO_NULL, fh, ierr)
    call MPI_FILE_SET_SIZE(fh, 0_MPI_OFFSET_KIND, ierr)
    call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
         MPI_ORDER_FORTRAN, MPI_REAL_RP, type_plane, ierr)
    call MPI_TYPE_COMMIT(type_plane, ierr)
    disp = 0_MPI_OFFSET_KIND
    call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL_RP, type_plane, 'native', MPI_INFO_NULL, ierr)
    call MPI_FILE_WRITE_ALL(fh, plane_read, product(subsizes), MPI_REAL_RP, MPI_STATUS_IGNORE, ierr)
    call MPI_TYPE_FREE(type_plane, ierr)
    call MPI_FILE_CLOSE(fh, ierr)
    if(myid == 0) print*, '*** [TEST] Read-back slice written to: ', trim(fname_test)

    deallocate(plane_read)

    if(myid == 0) print*, '*** [TEST] Complete! Compare files to verify: ***'
    if(myid == 0) print*, '    cmp ', trim(fname_orig), ' ', trim(fname_test)

  end subroutine test_2d_slice_io

end module mod_test_2d_io
