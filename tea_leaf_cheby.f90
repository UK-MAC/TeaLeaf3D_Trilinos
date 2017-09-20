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

!>  @brief Fortran heat conduction kernel
!>  @author Michael Boulton, Wayne Gaudin
!>  @details Implicitly calculates the change in temperature using accelerated Chebyshev method

MODULE tea_leaf_kernel_cheby_module

USE tea_leaf_kernel_common_module

IMPLICIT NONE

CONTAINS

SUBROUTINE tea_leaf_kernel_cheby_init(x_min,   &
                           x_max,              &
                           y_min,              &
                           y_max,              &
                           z_min,              &
                           z_max,              &
                           halo_exchange_depth,&
                           u,                  &
                           u0,                 &
                           p,                  &
                           r,                  &
                           Mi,                 &
                           w,                  &
                           z,                  &
                           Kx,                 &
                           Ky,                 &
                           Kz,                 &
                           cp,                 &
                           bfp,                &
                           rx,                 &
                           ry,                 &
                           rz,                 &
                           theta,              &
                           preconditioner_type)
  IMPLICIT NONE

  INTEGER :: preconditioner_type
  INTEGER(KIND=4):: x_min,x_max,y_min,y_max,z_min,z_max,halo_exchange_depth
  REAL(KIND=8), DIMENSION(x_min-halo_exchange_depth:x_max+halo_exchange_depth,&
                          y_min-halo_exchange_depth:y_max+halo_exchange_depth,&
                          z_min-halo_exchange_depth:z_max+halo_exchange_depth) :: u, u0, p , w , r, Mi, z , Kx, Ky, Kz
  REAL(KIND=8), DIMENSION(x_min:x_max,y_min:y_max,z_min-halo_exchange_depth:z_max+halo_exchange_depth) :: cp, bfp

  INTEGER :: j,k,l
  REAL(KIND=8) ::  rx, ry, rz, theta

!$OMP PARALLEL
!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
        DO j=x_min,x_max
            w(j, k, l) = (1.0_8                                             &
                + rx*(Kx(j+1, k, l) + Kx(j, k, l))                          &
                + ry*(Ky(j, k+1, l) + Ky(j, k, l))                          &
                + rz*(Kz(j, k, l+1) + Kz(j, k, l)))*u(j, k, l)              &
                - rx*(Kx(j+1, k, l)*u(j+1, k, l) + Kx(j, k, l)*u(j-1, k, l))&
                - ry*(Ky(j, k+1, l)*u(j, k+1, l) + Ky(j, k, l)*u(j, k-1, l))&
                - rz*(Kz(j, k, l+1)*u(j, k, l+1) + Kz(j, k, l)*u(j, k, l-1))
            r(j, k, l) = u0(j, k, l) - w(j, k, l)
        ENDDO
    ENDDO
  ENDDO
!$OMP END DO

  IF (preconditioner_type .NE. TL_PREC_NONE) THEN
    ! Use a block Jacobi preconditioner
    IF (preconditioner_type .EQ. TL_PREC_JAC_BLOCK) THEN
      CALL tea_block_solve(x_min, x_max, y_min, y_max, z_min, z_max, halo_exchange_depth, r, z, cp,&
                           bfp, Kx, Ky, Kz, rx, ry, rz)
    ! Use a point diagonal Jacobi preconditioner                       
    ELSE IF (preconditioner_type .EQ. TL_PREC_JAC_DIAG) THEN
      CALL tea_diag_solve(x_min, x_max, y_min, y_max, z_min, z_max, halo_exchange_depth, r, z, Mi)
    ENDIF

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        p(j, k, l) = z(j, k, l)/theta
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO NOWAIT
  ELSE
!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
        DO j=x_min,x_max
            p(j, k, l) = r(j, k, l)/theta
        ENDDO
    ENDDO
  ENDDO
!$OMP END DO NOWAIT
  ENDIF
!$OMP DO
  DO l=z_min,z_max
  DO k=y_min,y_max
      DO j=x_min,x_max
          u(j, k, l) = u(j, k, l) + p(j, k, l)
      ENDDO
  ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE

SUBROUTINE tea_leaf_kernel_cheby_iterate(x_min, &
                           x_max,               &
                           y_min,               &
                           y_max,               &
                           z_min,               &
                           z_max,               &
                           halo_exchange_depth, &
                           u,                   &
                           u0,                  &
                           p,                   &
                           r,                   &
                           Mi,                  &
                           w,                   &
                           z,                   &
                           Kx,                  &
                           Ky,                  &
                           Kz,                  &
                           cp,                  &
                           bfp,                 &
                           ch_alphas,           &
                           ch_betas,            &
                           max_cheby_iters,     &
                           rx,                  &
                           ry,                  &
                           rz,                  &
                           step,                &
                           preconditioner_type)

  IMPLICIT NONE

  INTEGER :: preconditioner_type
  INTEGER(KIND=4):: x_min,x_max,y_min,y_max,z_min,z_max,halo_exchange_depth
  REAL(KIND=8), DIMENSION(x_min-halo_exchange_depth:x_max+halo_exchange_depth,&
                          y_min-halo_exchange_depth:y_max+halo_exchange_depth,&
                          z_min-halo_exchange_depth:z_max+halo_exchange_depth) :: u, u0, p , w , r, Mi, z , Kx, Ky, Kz
  REAL(KIND=8), DIMENSION(x_min:x_max,y_min:y_max,z_min:z_max):: cp, bfp

  INTEGER :: j,k,l

    REAL(KIND=8) ::  rx, ry, rz

    INTEGER :: step, max_cheby_iters
    REAL(KIND=8), DIMENSION(max_cheby_iters) :: ch_alphas, ch_betas

!$OMP PARALLEL
!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
        DO j=x_min,x_max
            w(j, k, l) = (1.0_8                                             &
                + rx*(Kx(j+1, k, l) + Kx(j, k, l))                          &
                + ry*(Ky(j, k+1, l) + Ky(j, k, l))                          &
                + rz*(Kz(j, k, l+1) + Kz(j, k, l)))*u(j, k, l)              &
                - rx*(Kx(j+1, k, l)*u(j+1, k, l) + Kx(j, k, l)*u(j-1, k, l))&
                - ry*(Ky(j, k+1, l)*u(j, k+1, l) + Ky(j, k, l)*u(j, k-1, l))&
                - rz*(Kz(j, k, l+1)*u(j, k, l+1) + Kz(j, k, l)*u(j, k, l-1))
            r(j, k, l) = u0(j, k, l) - w(j, k, l)
        ENDDO
    ENDDO
  ENDDO
!$OMP END DO

  IF (preconditioner_type .NE. TL_PREC_NONE) THEN
    ! Use a block Jacobi preconditioner
    IF (preconditioner_type .EQ. TL_PREC_JAC_BLOCK) THEN
      CALL tea_block_solve(x_min, x_max, y_min, y_max, z_min, z_max, halo_exchange_depth, r, z, cp,&
                           bfp, Kx, Ky, Kz, rx, ry, rz)
    ! Use a point diagonal Jacobi preconditioner
    ELSE IF (preconditioner_type .EQ. TL_PREC_JAC_DIAG) THEN
      CALL tea_diag_solve(x_min, x_max, y_min, y_max, z_min, z_max, halo_exchange_depth, r, z, Mi)
    ENDIF

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        p(j, k, l) = ch_alphas(step)*p(j, k, l) + ch_betas(step)*z(j, k, l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO NOWAIT
  ELSE
!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        p(j, k, l) = ch_alphas(step)*p(j, k, l) + ch_betas(step)*r(j, k, l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO NOWAIT
  ENDIF
!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
        DO j=x_min,x_max
            u(j, k, l) = u(j, k, l) + p(j, k, l)
        ENDDO
    ENDDO
  ENDDO
!$OMP END DO NOWAIT
!$OMP END PARALLEL

END SUBROUTINE tea_leaf_kernel_cheby_iterate

SUBROUTINE tqli(d,e,n, info)
    ! http://physics.sharif.edu/~jafari/fortran-codes/lanczos/tqli.f90
    IMPLICIT NONE
    REAL(KIND=8), DIMENSION(n) :: d,e
    INTEGER :: i,iter,l,m,n,info
    REAL(KIND=8) :: b,c,dd,f,g,p,r,s
    e(:)=eoshift(e(:),1)
    info = 0
    DO l=1,n
      iter=0
      iterate: DO
        DO m=l,n-1
          dd=ABS(d(m))+ABS(d(m+1))
          IF (ABS(e(m))+dd == dd) EXIT
        ENDDO
        IF (m == l) EXIT iterate
        IF (iter == 30) THEN
          info=1
          RETURN
        ENDIF
        iter=iter+1
        g=(d(l+1)-d(l))/(2.0_8*e(l))
        r=SQRT(g**2.0_8+1.0_8**2.0_8)
        g=d(m)-d(l)+e(l)/(g+SIGN(r,g))
        s=1.0_8
        c=1.0_8
        p=0.0_8
        DO i=m-1,l,-1
          f=s*e(i)
          b=c*e(i)
          r=SQRT(f**2.0_8+g**2.0_8)
          e(i+1)=r
          IF (r == 0.0_8) THEN
            d(i+1)=d(i+1)-p
            e(m)=0.0_8
            CYCLE iterate
          ENDIF
          s=f/r
          c=g/r
          g=d(i+1)-p
          r=(d(i)-g)*s+2.0_8*c*b
          p=s*r
          d(i+1)=g+p
          g=c*r-b
        ENDDO
        d(l)=d(l)-p
        e(l)=g
        e(m)=0.0_8
      END DO iterate
    END DO
END SUBROUTINE tqli

SUBROUTINE tea_calc_eigenvalues(cg_alphas, cg_betas, eigmin, eigmax, max_iters, tl_ch_cg_presteps, info)

  INTEGER :: tl_ch_cg_presteps, max_iters
  REAL(KIND=8), DIMENSION(max_iters) :: cg_alphas, cg_betas
  REAL(KIND=8), DIMENSION(tl_ch_cg_presteps) :: diag, offdiag
  ! z not used for this
  REAL(KIND=8) :: eigmin, eigmax, tmp
  INTEGER :: n, info
  LOGICAL :: swapped

  diag = 0
  offdiag = 0

  DO n=1,tl_ch_cg_presteps
    diag(n) = 1.0_8/cg_alphas(n)
    IF (n .GT. 1) diag(n) = diag(n) + cg_betas(n-1)/cg_alphas(n-1)
    IF (n .LT. tl_ch_cg_presteps) offdiag(n+1) = SQRT(cg_betas(n))/cg_alphas(n)
  ENDDO

  CALL tqli(diag, offdiag, tl_ch_cg_presteps, info)

  ! could just call this instead
  !offdiag(:)=eoshift(offdiag(:),1)
  !CALL dsterf(tl_ch_cg_presteps, diag, offdiag, info)

  IF (info .NE. 0) RETURN

  ! bubble sort eigenvalues
  DO
    DO n=1,tl_ch_cg_presteps-1
      IF (diag(n) .GE. diag(n+1)) THEN
        tmp = diag(n)
        diag(n) = diag(n+1)
        diag(n+1) = tmp
        swapped = .TRUE.
      ENDIF
    ENDDO
    IF (.NOT. swapped) EXIT
    swapped = .FALSE.
  ENDDO

  eigmin = diag(1)
  eigmax = diag(tl_ch_cg_presteps)

  IF (eigmin .LT. 0.0_8 .OR. eigmax .LT. 0.0_8) info = 1

END SUBROUTINE tea_calc_eigenvalues

SUBROUTINE tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
    theta, max_cheby_iters)

  INTEGER :: n, max_cheby_iters
  REAL(KIND=8), DIMENSION(max_cheby_iters) :: ch_alphas, ch_betas
  REAL(KIND=8) :: eigmin, eigmax

  REAL(KIND=8) :: theta, delta, sigma, rho_old, rho_new, cur_alpha, cur_beta

  theta = (eigmax + eigmin)/2
  delta = (eigmax - eigmin)/2
  sigma = theta/delta

  rho_old = 1.0_8/sigma

  DO n=1,max_cheby_iters
    rho_new = 1.0_8/(2.0_8*sigma - rho_old)

    cur_alpha = rho_new*rho_old
    cur_beta = 2.0_8*rho_new/delta

    ch_alphas(n) = cur_alpha
    ch_betas(n) = cur_beta

    rho_old = rho_new
  ENDDO

END SUBROUTINE tea_calc_ch_coefs

END MODULE
