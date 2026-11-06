! Stapleton-Viaux exponential model with combined-max coupling.
! Purpose: Abaqus-style UMAT for the SSMCCF wrapper.
! Notes:
!   - Purely anisotropic model: no Neo-Hookean or isotropic matrix term.
!   - Nonlinear fiber stress uses saturating exponentials in I4 and I6.
!   - Coupling is K_coup*MAX(a*(I4-C1)+b*(I6-C2),0)**zeta.
!   - PROPS order is documented above Fasermaterial.

! Lightweight wrapper used by SSMCCF/f2py-style calls.
! It initializes Abaqus UMAT arguments that are not supplied directly.
      subroutine myumat(stress,props, dfgrd1,nprops)
      implicit none
      integer ntens
      integer nprops, nstatv
      real*8 ::  STRESS(6)
      real*8 :: STRAN(6), statev(10), DSTRAN(6)
      real*8 ::  DDSDDE(6,6),Props(nprops),STRANMAT(3,3)
      real*8  dfgrd1(3,3)
      integer:: i
      character*80 cmname
      integer ndi,nshr,noel,npt
      integer layer,kspt,kstep,kinc
      real*8 theta1, theta2
      real*8  sse,spd,scd,rpl,ddsddt(1),drplde(1)
      real*8  drpldt,time(2),dtime,dfgrd0(3,3)
      real*8  temp,dtemp,predef(3),dpred(3)
      real*8  coords(3),drot(3,3),pnewdt,celent
      do i=1,6
      STRAN(i)  =  0.0d0
      DSTRAN(i) =  0.0d0
      STRESS(i) =  0.0d0
      end do
      do i=1,10
      STATEV(i) = 0.0d0
      end do
      call umat(stress,statev,ddsdde,sse,spd,scd,
     &  rpl,ddsddt,drplde,drpldt,
     &  stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,
     &  ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt,
     &  celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc)
      return
      end

! Abaqus UMAT entry point.
! Input deformation gradient DFGRD1 is converted to C = F^T F.
! Output stress is Cauchy stress in Abaqus Voigt order.
      subroutine umat(stress,statev,ddsdde,sse,spd,scd,
     &  rpl,ddsddt,drplde,drpldt,
     &  stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,
     &  ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt,
     &  celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc)
      implicit none
      integer nprecd
      character*80 cmname
      integer ndi,nshr,ntens,nstatv,nprops,noel,npt
      integer layer,kspt,kstep,kinc
      real*8  stress(ntens),statev(nstatv),ddsdde(ntens,ntens)
      real*8  sse,spd,scd,rpl,ddsddt(ntens),drplde(ntens)
      real*8  drpldt,stran(ntens),dstran(ntens),time(2),dtime
      real*8  temp,dtemp,predef(1),dpred(1),props(nprops)
      real*8  coords(3),pnewdt,celent,dfgrd0(3,3), drot(3,3)
      real*8  dfgrd1(3,3)
      real*8 n1ref(3), n2ref(3)
      real*8 RCG(3,3), SPK(3,3), CaSi(3,3)
      real*8 dSdE(3,3,3,3)
      real*8 detF, theta1, theta2
      ddsddt=0.0d0
      n1ref=0.0d0
      n2ref=0.0d0
      RCG=0.0d0
      SPK=0.0d0
      CaSi=0.0d0
      dSdE=0.0d0
      detF=0.0d0
      theta1=0.0d0
      theta2=90.0d0

! Fiber directions are prescribed by two in-plane angles in degrees.
! They are kept in the reference configuration for invariant evaluation.
      theta1=props(11) * 3.1415927d0 / 180.0d0
      theta2=props(12) * 3.1415927d0 / 180.0d0
      n1ref = [dcos(theta1), dsin(theta1), 0.d0]
      n2ref = [dcos(theta2), dsin(theta2), 0.d0]

      call multMaTMa(dfgrd1,dfgrd1,RCG,3,3,3)
      call determinant33(dfgrd1,detF)

! Evaluate second Piola-Kirchhoff stress and material tangent dS/dE.
      call Fasermaterial(props, RCG, n1ref, n2ref, SPK, dSdE)

! Convert stress/tangent from material form to Abaqus Cauchy/Voigt form.
      call spk_to_cauchy (detF, dfgrd1, SPK, CaSi)
      call mat33n6_ABAQ(CaSi,stress)
      call dSdE_to_dSigdEps(dfgrd1,detF, SPK,dSdE,ddsdde)
      return
      end


! Constitutive model in terms of C = F^T F and fiber directions N1, N2.
! PROPS order:
!   1 mu1, 2 mu2
!   3 K_1ani1, 4 beta_1, 5 K_1ani2, 6 gamma_1
!   7 K_coup, 8 zeta, 9 C1, 10 C2
!   11 theta1, 12 theta2, 13 mu3, 14 a, 15 b
      subroutine Fasermaterial(d, RCG, N1, N2, SPK, dSdE)
      implicit none
      integer i, j, k, l
      real*8 N1(3), N2(3), d(*), RCG(3,3)
      real*8 SPK(3,3), dSdE(3,3,3,3)
      real*8 mu, mu2, mu3
      real*8 K_1ani1, beta_1
      real*8 K_1ani2, gamma_1
      real*8 K_coup, zeta, a_coup, b_coup
      real*8 Mtens1(3,3), Mtens2(3,3)
      real*8 C1, C2
      real*8 I4, I6
      real*8 expterm4, expterm6
      real*8 d2WdI4dI4, d2WdI6dI6
      real*8 dI4dC(3,3), dI6dC(3,3)
      real*8 dWdI4, dWdI6
      real*8 coup_arg
      real*8 dWcoupdI4, dWcoupdI6
      real*8 d2Wcoup44, d2Wcoup66, d2Wcoup46
      real*8 SPKlin(3,3), SPKani(3,3)
      real*8 I6help(3), I4help(3)
      mu=d(1)
      mu2=d(2)
      K_1ani1=d(3)
      beta_1=d(4)
      K_1ani2=d(5)
      gamma_1=d(6)
      K_coup=d(7)
      zeta=d(8)
      C1=d(9)
      C2=d(10)
      mu3=d(13)
! Coupling weights; example: a_coup=b_coup=0.5d0.
      a_coup=d(14)
      b_coup=d(15)

      call dyprod3_BS(n1,n1,Mtens1)
      call dyprod3_BS(n2,n2,Mtens2)

! Fiber invariants of C and their first derivatives.
      call multmavec3_BS(RCG,N1,I4help)
      call multvecvec3_BS(N1,I4help,I4)
      call multmavec3_BS(RCG,N2,I6help)
      call multvecvec3_BS(N2,I6help,I6)
      do i = 1,3
        do j = 1,3

! Stress contributions: linear anisotropic and nonlinear fiber terms.
          SPKlin(i,j) =   4.0d0*mu*(I4-1)*Mtens1(i,j)
     &               + 4.0d0*mu2*(I6-1)*Mtens2(i,j)
     &               + 2.0d0*mu3*((I6-1)*Mtens1(i,j)+(I4-1)*Mtens2(i,j))
        end do
      end do
      dI4dC=Mtens1
      dI6dC=Mtens2
      if(I4.le.C1)then
      K_1ani1=0.d0
      endif
      if(I6.le.C2)then
      K_1ani2=0.d0
      endif

      expterm4=EXP(-beta_1*(I4-1.d0))
      expterm6=EXP(-gamma_1*(I6-1.d0))
      d2WdI4dI4=K_1ani1*beta_1*expterm4
      d2WdI6dI6=K_1ani2*gamma_1*expterm6

! Activate coupling with the positive part of the weighted excess.
      coup_arg=MAX(a_coup*(I4-C1)+b_coup*(I6-C2),0.d0)
      dWcoupdI4=0.d0
      dWcoupdI6=0.d0
      d2Wcoup44=0.d0
      d2Wcoup66=0.d0
      d2Wcoup46=0.d0
! Both derivatives activate when the combined argument is positive.
      if(coup_arg.gt.0.d0)then
        dWcoupdI4=K_coup*zeta*a_coup*
     &                 coup_arg**(zeta-1.d0)
        dWcoupdI6=K_coup*zeta*b_coup*
     &                 coup_arg**(zeta-1.d0)
        d2Wcoup44=K_coup*zeta*(zeta-1.d0)*a_coup*a_coup*
     &                 coup_arg**(zeta-2.d0)
        d2Wcoup66=K_coup*zeta*(zeta-1.d0)*b_coup*b_coup*
     &                 coup_arg**(zeta-2.d0)
        d2Wcoup46=K_coup*zeta*(zeta-1.d0)*a_coup*b_coup*
     &                 coup_arg**(zeta-2.d0)
      endif

      dWdI4=  K_1ani1*(1.d0-expterm4)
     &        +dWcoupdI4
      dWdI6=  K_1ani2*(1.d0-expterm6)
     &        +dWcoupdI6
      SPKani=2.0d0 * (dWdI4*dI4dC + dWdI6*dI6dC)
      SPK = SPKlin + SPKani
      do i=1,3
        do j=1,3
          do k=1,3
            do l=1,3
            dSdE(i,j,k,l)=0.d0
            end do
          end do
        end do
      end do
      do i=1,3
        do j=1,3
          do k=1,3
            do l=1,3
      dSdE(i,j,k,l)=2.d0*mu*MTens1(i,j)*MTens1(k,l)
     &      +2.d0*mu2*MTens2(i,j)*MTens2(k,l)
     &      +mu3*(MTens2(i,j)*MTens1(k,l)+MTens1(i,j)*MTens2(k,l))
      dSdE(i,j,k,l)=dSdE(i,j,k,l)
     &         +d2WdI4dI4*MTens1(i,j)*MTens1(k,l)
      dSdE(i,j,k,l)=dSdE(i,j,k,l)
     &         +d2WdI6dI6*MTens2(i,j)*MTens2(k,l)
      dSdE(i,j,k,l)=dSdE(i,j,k,l)
     &         +(d2Wcoup44*MTens1(i,j)*MTens1(k,l)
     &         +d2Wcoup66*MTens2(i,j)*MTens2(k,l)
     &         +d2Wcoup46*
     &         (MTens2(i,j)*MTens1(k,l)+MTens1(i,j)*MTens2(k,l)))
      dSdE(i,j,k,l)=4.d0*dSdE(i,j,k,l)
            end do
          end do
        end do
      end do
      return
      end

! Tensor utility routines used by the UMAT.
      subroutine spk_to_cauchy (detF, F, spk, CaSi)
      implicit none
      real*8 F(3,3), detF, spk(3,3), CaSi(3,3)
      integer i, j, a, b
      do a=1,3
        do b=1,3
          CaSi(a,b)=0.d0
          do i=1,3
            do j=1,3
              CaSi(a,b)=CaSi(a,b)+F(a,i)*F(b,j)*spk(i,j)/detF
            end do
          end do
        end do
      end do
      return
      end
      subroutine dyprod3_BS (V1,V2,M)
      implicit none
      real*8 V1(3), V2(3), M(3,3)
      M(1,1)=V1(1)*V2(1)
      M(1,2)=V1(1)*V2(2)
      M(1,3)=V1(1)*V2(3)
      M(2,1)=V1(2)*V2(1)
      M(2,2)=V1(2)*V2(2)
      M(2,3)=V1(2)*V2(3)
      M(3,1)=V1(3)*V2(1)
      M(3,2)=V1(3)*V2(2)
      M(3,3)=V1(3)*V2(3)
      return
      end
      subroutine multmavec3_BS (M,V,U)
      implicit none
      real*8 M(3,3), V(3), U(3)
      U(1)=M(1,1)*V(1)+M(1,2)*V(2)+M(1,3)*V(3)
      U(2)=M(2,1)*V(1)+M(2,2)*V(2)+M(2,3)*V(3)
      U(3)=M(3,1)*V(1)+M(3,2)*V(2)+M(3,3)*V(3)
      return
      end
      subroutine multvecvec3_BS (V1,V2,skalar)
      implicit none
      real*8 V1(3), V2(3), skalar
      skalar=V1(1)*V2(1)+V1(2)*V2(2)+V1(3)*V2(3)
      return
      end
      subroutine mat33n6_ABAQ(b9,a6)
      implicit none
      integer i,j
      real*8  a6(6),b9(3,3)
      do i=1,3
      a6(i)=b9(i,i)
      end do
      a6(4)=b9(1,2)
      a6(5)=b9(1,3)
      a6(6)=b9(2,3)
      return
      end
      subroutine vier_zweiABAQ(d,dss66)
      implicit none
      integer i
      real*8  dss66(6,6),d(3,3,3,3)
      do i=1,3
      dss66(i,1)=d(i,i,1,1)
      dss66(i,2)=d(i,i,2,2)
      dss66(i,3)=d(i,i,3,3)
      dss66(i,4)=d(i,i,1,2)
      dss66(i,5)=d(i,i,1,3)
      dss66(i,6)=d(i,i,2,3)
      end do
      dss66(4,1)=d(1,2,1,1)
      dss66(4,2)=d(1,2,2,2)
      dss66(4,3)=d(1,2,3,3)
      dss66(4,4)=d(1,2,1,2)
      dss66(4,5)=d(1,2,1,3)
      dss66(4,6)=d(1,2,2,3)
      dss66(5,1)=d(1,3,1,1)
      dss66(5,2)=d(1,3,2,2)
      dss66(5,3)=d(1,3,3,3)
      dss66(5,4)=d(1,3,1,2)
      dss66(5,5)=d(1,3,1,3)
      dss66(5,6)=d(1,3,2,3)
      dss66(6,1)=d(2,3,1,1)
      dss66(6,2)=d(2,3,2,2)
      dss66(6,3)=d(2,3,3,3)
      dss66(6,4)=d(2,3,1,2)
      dss66(6,5)=d(2,3,1,3)
      dss66(6,6)=d(2,3,2,3)
      return
      end
      subroutine determinant33(mat,det)
      implicit none
      real*8, intent(in) ::  mat(3,3)
      real*8, intent(inout) :: det
      det= mat(1,1)*mat(2,2)*mat(3,3)
     &     + mat(1,2)*mat(2,3)*mat(3,1)
     &     + mat(1,3)*mat(2,1)*mat(3,2)
     &     - mat(1,1)*mat(2,3)*mat(3,2)
     &     - mat(2,2)*mat(3,1)*mat(1,3)
     &     - mat(3,3)*mat(1,2)*mat(2,1)
      return
      end
      subroutine idendityMatrix(ident, n)
      implicit none
      integer n, i, j
      real*8 ident(n,n)
      do i=1,n
       do j=1,n
        ident(i,j)=0.d0
       end do
       ident(i,i) = 1.d0
      end do
      return
      end
      subroutine multMaMa(a,b,c,l,m,n)
      implicit none
      integer l,m,n,i,j,k
      real*8 a(l,m),b(m,n),c(l,n)
      do i=1,l
      do j=1,n
      c(i,j)=0.d0
      do k=1,m
            c(i,j)=c(i,j)+a(i,k)*b(k,j)
      end do
      end do
      end do
      return
      end
      subroutine multMaTMa(a,b,c,l,m,n)
      implicit none
      integer l,m,n,i,j,k
      real*8 a(m,l),b(m,n),c(l,n)
      do i=1,l
      do j=1,n
      c(i,j)=0.d0
      do k=1,m
            c(i,j)=c(i,j)+a(k,i)*b(k,j)
      end do
      end do
      end do
      return
      end
      subroutine dSdE_to_dSigdEps(dfgrd1,detF,spk,dSdE,dSigdEps_Abaqus)
      implicit none

      double precision, intent(in) :: dfgrd1(3,3)
      double precision, intent(in) :: spk(3,3)
      double precision, intent(in) :: detF
      double precision, intent(in) :: dSdE(3,3,3,3)
      double precision, intent(out) :: dSigdEps_Abaqus(6,6)

      integer :: i,j,k,l,m,n
      double precision :: fp33(3,3)
      double precision :: on33(3,3)
      double precision :: A_4(3,3,3,3)
      double precision :: T_4(3,3,3,3)
      double precision :: Tsum4(3,3,3,3)
      double precision :: dSigdEps_Tens(6,6)


      on33      = 0.0d0
      on33(1,1) = 1.0d0
      on33(2,2) = 1.0d0
      on33(3,3) = 1.0d0

c
c.... Push-forward of the tangent
c
      do l=1,3
          do k=1,3
            do j=1,3
              do i=1,3
                A_4(i,j,k,l)=0.d0
                T_4(i,j,k,l)=0.d0
                Tsum4(i,j,k,l)=0.d0
              end do
            end do
          end do
        end do
c
      do i=1,3
          do j=1,3
            do k=1,3
              do l=1,3
                A_4(i,j,k,l)=on33(i,k)*spk(j,l)
                do m=1,3
                  do n=1,3
                    A_4(i,j,k,l)=A_4(i,j,k,l)
     &                +dfgrd1(i,m)*dSdE(m,j,n,l)*dfgrd1(k,n)
                  end do
                end do
              end do
            end do
          end do
        end do
c
      call multmama(dfgrd1,spk,fp33,3,3,3)
c
      do i=1,3
          do j=1,3
            do k=1,3
              do l=1,3
                do m=1,3
                    T_4(i,j,k,l)=T_4(i,j,k,l)+A_4(i,m,k,l)*dfgrd1(j,m)
     &                +fp33(i,m)*on33(j,k)*on33(m,l)
                end do
              end do
            end do
          end do
        end do
c
      do i=1,3
          do j=1,3
            do k=1,3
              do l=1,3
                do m=1,3
                 Tsum4(i,j,k,l)=Tsum4(i,j,k,l)+(T_4(i,j,k,m)*dfgrd1(l,m)
     &                +T_4(i,j,l,m)*dfgrd1(k,m))/2.d0
                end do
                Tsum4(i,j,k,l)=Tsum4(i,j,k,l)/detF
              end do
            end do
          end do
        end do
c
c
       call vier_zweiABAQ(Tsum4,dSigdEps_Abaqus)
      return
      end subroutine dSdE_to_dSigdEps
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc]
