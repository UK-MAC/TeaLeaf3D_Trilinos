!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Holds the high level Fortran data types
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The high level data types used to store the mesh and field data
!>  are defined here.
!>
!>  Also the global variables used for defining the input and controlling the
!>  scheme are defined here.

MODULE definitions_module

   USE data_module

   IMPLICIT NONE

   TYPE state_type
      LOGICAL            :: defined

      REAL(KIND=8)       :: density          &
                           ,energy

      INTEGER            :: geometry

      REAL(KIND=8)       :: xmin               &
                           ,xmax               &
                           ,ymin               &
                           ,ymax               &
                           ,zmin               &
                           ,zmax               &
                           ,radius
   END TYPE state_type

   TYPE(state_type), ALLOCATABLE             :: states(:)
   INTEGER                                   :: number_of_states

   TYPE grid_type
     REAL(KIND=8)       :: xmin            &
                          ,ymin            &
                          ,zmin            &
                          ,xmax            &
                          ,ymax            &
                          ,zmax
                     
     INTEGER            :: x_cells              &
                          ,y_cells              &
                          ,z_cells
   END TYPE grid_type

   INTEGER      :: step

   INTEGER      :: error_condition

   INTEGER      :: test_problem
   LOGICAL      :: complete

   LOGICAL      :: use_fortran_kernels
   LOGICAL      :: tl_use_chebyshev
   LOGICAL      :: tl_use_cg
   LOGICAL      :: tl_use_ppcg
   LOGICAL      :: tl_use_jacobi
   
   ! For Trilinos
   LOGICAL      :: use_trilinos_kernels


   INTEGER      :: total_petsc_iter
   INTEGER      :: solver_type   
   
   LOGICAL      :: verbose_on
   INTEGER      :: max_iters
   REAL(KIND=8) :: eps
   INTEGER      :: coefficient

   ! error to run cg to before calculating eigenvalues
   REAL(KIND=8) :: tl_ch_cg_epslim
   
   ! number of steps of cg to run to before switching to ch if tl_ch_cg_errswitch not set
   INTEGER      :: tl_ch_cg_presteps
   
   ! do b-Ax after finishing to make sure solver actually converged
   LOGICAL      :: tl_check_result
   
   ! number of inner steps in ppcg solver
   INTEGER      :: tl_ppcg_inner_steps
   
   ! activate ppcg  (true), deactivate (false)
   LOGICAL      :: tl_ppcg_active

   ! Reflective boundaries at edge of mesh
   LOGICAL      :: reflective_boundary

   ! Preconditioner option
   INTEGER      :: tl_preconditioner_type

   LOGICAL      :: use_vector_loops ! Some loops work better in serial depending on the hardware

   LOGICAL      :: profiler_on ! Internal code profiler to make comparisons across systems easier

   TYPE profiler_type
     REAL(KIND=8)       :: timestep        &
                          ,visit           &
                          ,summary         &
                          ,tea_init        &
                          ,tea_solve       &
                          ,tea_reset       &
                          ,set_field       &
                          ,dot_product     &
                          ,halo_update     &
                          ,halo_exchange

   END TYPE profiler_type
   TYPE(profiler_type)  :: profiler

   REAL(KIND=8) :: end_time

   INTEGER      :: end_step

   REAL(KIND=8) :: dt             &
                  ,time           &
                  ,dtinit

   INTEGER      :: visit_frequency   &
                  ,summary_frequency

   INTEGER         :: jdt,kdt

   TYPE field_type
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: density
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: energy0,energy1
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: u, u0
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_p
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_r
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_rstore
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_rtemp
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_utemp

     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_Mi
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_w
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_z
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_Kx
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_Ky
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_sd
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: tri_cp, tri_bfp
     REAL(KIND=8),    DIMENSION(:,:,:), ALLOCATABLE :: vector_Kz

     INTEGER         :: left            &
                       ,right           &
                       ,bottom          &
                       ,top             &
                       ,back            &
                       ,front           &
                       ,left_boundary   &
                       ,right_boundary  &
                       ,bottom_boundary &
                       ,top_boundary    &
                       ,front_boundary  &
                       ,back_boundary

     INTEGER         :: x_min  &
                       ,y_min  &
                       ,z_min  &
                       ,x_max  &
                       ,y_max  &
                       ,z_max

     REAL(KIND=8), DIMENSION(:),   ALLOCATABLE :: cellx    &
                                                 ,celly    &
                                                 ,cellz    &
                                                 ,vertexx  &
                                                 ,vertexy  &
                                                 ,vertexz  &
                                                 ,celldx   &
                                                 ,celldy   &
                                                 ,celldz   &
                                                 ,vertexdx &
                                                 ,vertexdy &
                                                 ,vertexdz

     REAL(KIND=8), DIMENSION(:,:,:), ALLOCATABLE :: volume  &
                                                 ,xarea     &
                                                 ,yarea     &
                                                 ,zarea

   END TYPE field_type

   TYPE chunk_type

     INTEGER         :: task   !mpi task

     INTEGER         :: chunk_neighbours(6) ! Chunks, not tasks, so we can overload in the future

     ! Idealy, create an array to hold the buffers for each field so a commuincation only needs
     !  one send and one receive per face, rather than per field.
     ! If chunks are overloaded, i.e. more chunks than tasks, might need to pack for a task to task comm
     !  rather than a chunk to chunk comm. See how performance is at high core counts before deciding
     REAL(KIND=8),ALLOCATABLE:: left_snd_buffer(:),right_snd_buffer(:)
     REAL(KIND=8),ALLOCATABLE:: bottom_snd_buffer(:),top_snd_buffer(:)
     REAL(KIND=8),ALLOCATABLE:: back_snd_buffer(:),front_snd_buffer(:)
     REAL(KIND=8),ALLOCATABLE:: left_rcv_buffer(:),right_rcv_buffer(:)
     REAL(KIND=8),ALLOCATABLE:: bottom_rcv_buffer(:),top_rcv_buffer(:)
     REAL(KIND=8),ALLOCATABLE:: back_rcv_buffer(:),front_rcv_buffer(:)

     TYPE(field_type):: field

  END TYPE chunk_type

  ! depth of halo for matrix powers
  integer :: halo_exchange_depth

  TYPE(chunk_type),  ALLOCATABLE       :: chunks(:)
  INTEGER                              :: number_of_chunks

  TYPE(grid_type)                      :: grid

END MODULE definitions_module
