*******************************************************************
	PROGRAM PROBLEMA I
*******************************************************************

	IMPLICIT NONE
	INTEGER i,j,N,NTS,NV,M,Ms,contP,contPmax,contS,contSmax,ierr
	REAL*8 erroP,erroU,erroS,erroG,convP,convU,convS,convG,Pini,Tini,
     &	   Pout,PM,CP,CV,VISC,dt,dze,tmax,error,L,D,epsilon,Gin,Tin,
     &	   Uin,Sin,Ginold
	REAL*8,ALLOCATABLE :: dz(:),P(:),P0(:),Pold(:),Paux(:),U(:),U0(:),
     &					  Uold(:),S(:),S0(:),Sold(:),T(:),RHO(:),VS(:),
     &					  BETA(:),f(:),A(:),K(:),X(:),AS(:),KS(:)

	
	CALL ENTRADAS(L,D,epsilon,tmax,dt,dze,convP,convU,convS,convG,
     &			  contPmax,contSmax,Pini,Tini,Pout,PM,CP,VISC,NV,M,Ms,
     &			  N,NTS)
	

	ALLOCATE(dz(N),P(N),P0(N),Pold(N),Paux(N),U(N),U0(N),Uold(N),S(N),
     &		 S0(N),Sold(N),T(N),RHO(N),VS(N),BETA(N),f(N),
     &		 A((2*M+1)*NV*N-M**2-M),K(NV*N),X(NV*N),
     &		 AS((2*Ms+1)*N-Ms**2-Ms),KS(N))


	dz = dze

	CALL INIT(Pini,Tini,P,U,S,T,RHO,VS,BETA,CP,CV,N,PM)

	CALL RESERVOIR(Gin,Tin,P(1))

	DO j=1,NTS

		write(*,*)j
c		pause

		P0 = P
		U0 = U
		S0 = S

		Ginold = Gin

		erroG = 1.D8
		DO WHILE(erroG.GT.convG)

			erroS = 1.D8
			contS = 0
			DO WHILE(erroS.GT.convS)

				contS = contS + 1
				Sold = S
				Paux = P

				erroP = 1.D3
				erroU = 1.D3
				contP = 0
				DO WHILE(erroP.GT.convP.OR.erroU.GT.convU)

					contP = contP + 1

					Pold = P
					Uold = U

					Uin = Gin/RHO(N)

					CALL Calcf(U,RHO,VISC,D,epsilon,f,N)

					CALL CalcA(U,RHO,VS,dz,dt,NV,N,M,A)

					CALL CalcK(U,Pout,Uin,P0,U0,f,D,dt,K,NV,N)

					CALL GBAND(A,K,X,NV*N,M,1.D-3,ierr,0)

					DO i=1,N
						P(i) = X((i-1)*NV+1)
						U(i) = X((i-1)*NV+2)
					END DO	
	
					DO i=1,N
						RHO(i) = RHO(i) + (P(i)-Pold(i))/VS(i)**2
					END DO

					erroP = 0.D0
					erroU = 0.D0

					DO i=1,N
						error = abs(P(i) - Pold(i))
						IF(error.GT.erroP) THEN
							erroP = error
							IF(erroP.GT.convP) GO TO 10
						END IF
					END DO

					DO i=1,N
						error = abs(U(i) - Uold(i))
						IF(error.GT.erroU) THEN
							erroU = error
							IF(erroU.GT.convU) GO TO 10
						END IF
					END DO

10					CONTINUE

					IF(contP.GE.contPmax) THEN
						WRITE(*,*) 'NUMERO DE ITERACOES EXCEDIDO'
						WRITE(*,*) 'erroP = ',erroP
						WRITE(*,*) 'erroU = ',erroU
						GO TO 11
					END IF
				END DO

11				CONTINUE

				Sin = CP*DLOG(Tin/300.D0) - 8.314D3/PM*DLOG(P(N)/101325.D0)

				CALL CalcAS(U,dz,dt,AS,Ms,N)

				CALL CalcKS(Sin,S0,dt,KS,N)

				CALL GBAND(AS,KS,S,N,Ms,1.D-3,ierr,0)


				CALL CalcPROP(P,T,S,RHO,CP,CV,BETA,VS,PM,N)


				erroS = 0.D0
				DO i=1,N
					error = abs(S(i) - Sold(i))
					IF(error.GT.erroS) THEN
						erroS = error
						IF(erroS.GT.convS) GO TO 14
					END IF
				END DO

14				CONTINUE

				IF(contS.GE.contSmax) THEN
					WRITE(*,*) 'NUMERO DE ITERACOES EXCEDIDO'
					WRITE(*,*) 'erroS = ',erroS
					GO TO 15
				END IF
			END DO

15			CONTINUE


			CALL RESERVOIR(Gin,Tin,P(1))
		
			erroG = DABS(Gin - Ginold)

		
		END DO		


	END DO

	CALL RESULTADOS(P,U,T,RHO,dz,N)

	END


*******************************************************************
	SUBROUTINE RESULTADOS(P,U,T,RHO,dz,N)
*******************************************************************

	IMPLICIT NONE
	INTEGER i,N
	REAL*8 P(N),U(N),T(N),RHO(N),dz(N),z		
	CHARACTER CONDICAO*30


91	FORMAT(D17.10,5X,D15.8,5X,D15.8,5X,D15.8,5X,D15.8,5X,D15.8)


	OPEN(UNIT=1, FILE='RESULTADOS.DAT', ACTION='WRITE', STATUS='UNKNOWN')

	dz(1) = 0.D0
	z = 0.D0
	DO i=1,N
		WRITE(1,91)P(i)/1.D3,U(i),T(i),RHO(i),z + dz(i)
		z = z + dz(i)
	END DO

	CLOSE(UNIT=1)


	RETURN
	END


************************************************************************
	SUBROUTINE ENTRADAS(L,D,epsilon,tmax,dt,dze,convP,convU,convS,
     &					convG,contPmax,contSmax,Pini,Tini,Pout,PM,CP,
     &					VISC,NV,M,Ms,N,NTS)
************************************************************************

	IMPLICIT NONE
	INTEGER i,N,NTS,NV,NVs,M,Ms,REF,contPmax,contSmax
	REAL*8 L,D,epsilon,tmax,dt,convP,convU,convS,convG,PM,CP,VISC,Pini,
     &	   Tini,Pout,dze


	OPEN(UNIT=1, FILE='DADOS.DAT', ACTION='READ', STATUS='OLD')


C	PARÂMETROS GLOBAIS
C	------------------
	READ(1,*)L			!comprimento da tubulação
	READ(1,*)D			!diâmetro da tubulação
	READ(1,*)epsilon	!rugosidade absoluta
	READ(1,*)tmax		!tempo de simulação


C	MALHA COMPUTACIONAL
C	-------------------
	READ(1,*)dt		!incremento de tempo
	READ(1,*)dze	!incremento de comprimento

	READ(1,*)convP		!convergência da pressão (máxima variação admissível)
	READ(1,*)convU		!convergência da velocidade (máxima variação admissível)
	READ(1,*)convS		!convergência da entropia (máxima variação admissível)
	READ(1,*)convG		!convergência do fluxo de massa na entrada (máxima variação admissível)
	READ(1,*)contPmax	!número máximo de vezes que o GBAND é chamado no loop interno 
	READ(1,*)contSmax	!número máximo de vezes que o GBAND é chamado no loop externo

	NTS = NINT(tmax/dt) + 1	!número de time-steps


C	GBAND
C	-----
	READ(1,*)NV		!número de variáveis resolvidas via GBAND no loop interno
	READ(1,*)NVs	!número de variáveis resolvidas via GBAND no loop externo

	M = ((6*NV-1) - 1)/2
	Ms = ((6*NVs-1) - 1)/2


C	FLUIDO
C	------

	READ(1,*)PM		!peso molecular do fluido
	READ(1,*)CP		!calor específico a pressão constante
	READ(1,*)VISC	!viscosidade do fluido
	READ(1,*)Pini	!pressão inicial
	READ(1,*)Tini	!temperatura inicial
	READ(1,*)Pout	!pressão na saída
	CLOSE(UNIT=1)


C	VARIÁVEIS ADICIONAIS
C	--------------------

	N = L/dze + 1		!número máximo de nós


	RETURN
	END


************************************************************************
	SUBROUTINE INIT(Pini,Tini,P,U,S,T,RHO,VS,BETA,CP,CV,N,PM)
************************************************************************

	IMPLICIT NONE
	INTEGER i,N
	REAL*8 Pini,Tini,P(N),U(N),S(N),T(N),RHO(N),VS(N),BETA(N),CP,
     &	   CV,PM,R


	R = 8.314D3/PM


	P = Pini

	U = 0.D0

	T = Tini

	RHO = Pini/(R*Tini)

	CP = 1005.D0

	CV = CP - R

	VS = DSQRT(CP/CV*R*Tini)
	
	BETA = 1.D0/Tini

	S = CP*DLOG(Tini/300.D0) - R*DLOG(Pini/101325.D0)


	RETURN
	END


************************************************************************
	SUBROUTINE CalcPROP(P,T,S,RHO,CP,CV,BETA,VS,PM,N)
************************************************************************

	IMPLICIT NONE
	INTEGER i,N
	REAL*8 P(N),T(N),S(N),RHO(N),CP,CV,BETA(N),VS(N),PM,R


	R = 8.314D3/PM

	DO i=1,N

		T(i) = 300.D0*DEXP(S(i)/CP)*(P(i)/101325.D0)**(R/CP)
		
		RHO(i) = P(i)/(R*T(i))

		VS(i) = DSQRT(CP/CV*R*T(i))

		BETA(i) = 1.D0/T(i)

	END DO


	RETURN
	END


************************************************************************
	SUBROUTINE CalcA(U,RHO,VS,dz,dt,NV,N,M,A)
************************************************************************

	IMPLICIT NONE
	INTEGER i,NV,N,NI,M,j,k,jmin,jmax,cont,x,CONT1,CONT2
	REAL*8 U(N),VS(N),RHO(N),dz(N),dt,A((2*M + 1)*NV*N - M**2 - M)
	REAL*8 G1(NV,NV),G2(NV,NV),B1,B2,M1,M2,F1,F2,B,F,BB1,BB2,FF1,FF2

	
	NI = NV*N

	A = 0.D0

	cont = 1
	DO i=1,NI

		IF(i.EQ.2) THEN
			A(cont+1) = 1.D0
			cont = cont + M + 2
			GO TO 20
		END IF

		IF(i.EQ.NI-1) THEN
			A(cont+5) = 1.D0
			cont = cont + M + 2
			GO TO 20
		END IF

		IF(i.LE.M) THEN
			jmin = 1
			jmax = i + M
		ELSE IF(i.GT.M.AND.i.LE.NI-M) THEN
			jmin = i - M
			jmax = jmin + 2*M
		ELSE
			jmin = i - M
			jmax = NI
		END IF

		k = (i-1)/NV + 1

		IF(i.EQ.(k-1)*NV+1) THEN
			G1(1,1) = (U(k)+VS(k))/2
			G1(1,2) = (U(k)+VS(k))*RHO(k)*VS(k)/2
			G1(2,1) = (U(k)+VS(k))/(2*RHO(k)*VS(k))
			G1(2,2) = (U(k)+VS(k))/2

			G2(1,1) = (U(k)-VS(k))/2
			G2(1,2) = -(U(k)-VS(k))*RHO(k)*VS(k)/2
			G2(2,1) = -(U(k)-VS(k))/(2*RHO(k)*VS(k))
			G2(2,2) = (U(k)-VS(k))/2
		END IF

		x = i - (k-1)*NV

		B = (dz(k) + dz(k-1))/dz(k)
		F = (dz(k+1) + dz(k+2))/dz(k+1)

		B1 = 0.D0
		B2 = 0.D0
		BB1 = 0.D0
		BB2 = 0.D0

		IF(i.GT.NV.AND.i.LE.2*NV) THEN
			B1 = -G1(x,1)/dz(k)
			B2 = -G1(x,2)/dz(k)
		ELSE IF(i.GT.2*NV) THEN
			B1 = -(B**2/(B**2-B))*G1(x,1)/dz(k)
			B2 = -(B**2/(B**2-B))*G1(x,2)/dz(k)
			BB1 = (1.D0/(B**2-B))*G1(x,1)/dz(k)
			BB2 = (1.D0/(B**2-B))*G1(x,2)/dz(k)
		END IF

		F1 = 0.D0
		F2 = 0.D0
		FF1 = 0.D0
		FF2 = 0.D0

		IF(i.GT.NI-2*NV.AND.i.LE.NI-NV) THEN
			F1 = G2(x,1)/dz(k+1)
			F2 = G2(x,2)/dz(k+1)
		ELSE IF(i.LE.NI-2*NV) THEN
			F1 = (F**2/(F**2-F))*G2(x,1)/dz(k+1)
			F2 = (F**2/(F**2-F))*G2(x,2)/dz(k+1)
			FF1 = -(1.D0/(F**2-F))*G2(x,1)/dz(k+1)
			FF2 = -(1.D0/(F**2-F))*G2(x,2)/dz(k+1)
		END IF

		M1 = -BB1 - B1 - F1 - FF1
		M2 = -BB2 - B2 - F2 - FF2

		IF(i.EQ.(k-1)*NV+1) THEN
			M1 = 1/dt + M1
		ELSE
			M2 = 1/dt + M2
		END IF

		DO j=jmin,jmax

			IF(j.EQ.(k-1)*NV-3) THEN
				A(cont) = BB1
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV-2) THEN
				A(cont) = BB2
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV-1) THEN
				A(cont) = B1
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV) THEN
				A(cont) = B2
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV+1) THEN
				A(cont) = M1
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV+2) THEN
				A(cont) = M2
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV+3) THEN
				A(cont) = F1
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV+4) THEN
				A(cont) = F2
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV+5) THEN
				A(cont) = FF1
				GO TO 10
			ELSE IF(j.EQ.(k-1)*NV+6) THEN
				A(cont) = FF2
				GO TO 10
			ELSE
				A(cont) = 0.D0
			END IF

10			CONTINUE

			cont = cont + 1
		END DO

20		CONTINUE

	END DO


	RETURN
	END


************************************************************************
	SUBROUTINE CalcK(U,Pout,Uin,P0,U0,f,D,dt,K,NV,N)
************************************************************************

	IMPLICIT NONE
	INTEGER i,j,x,NV,N,NI
	REAL*8 U(N),Pout,Uin,P0(N),U0(N),f(N),D,dt,K(NV*N),C(NV),Var0(NV)


	NI = NV*N

	K = 0.D0

	C(1) = 0.D0
	C(2) = -9.81D0

	DO i=1,NI

		IF(i.EQ.2) THEN
			K(i) = Uin
			GO TO 20
		END IF

		IF(i.EQ.NI-1) THEN
			K(i) = Pout
			GO TO 20
		END IF

		j = (i-1)/NV + 1

		IF(i.EQ.(j-1)*NV+1) THEN
			C(2) = -9.81D0 - f(j)*DABS(U(j))*U(j)/(2.D0*D)
c			C(2) = -9.81D0

			Var0(1) = P0(j)
			Var0(2) = U0(j)
		END IF

		x = i - (j-1)*NV

		K(i) = Var0(x)/dt + C(x)

20		CONTINUE

	END DO


	RETURN
	END


************************************************************************
	SUBROUTINE CalcAS(U,dz,dt,A,M,N)
************************************************************************

	IMPLICIT NONE
	INTEGER i,cont,N,j,jmin,jmax,M
	REAL*8 U(N),dz(N),A((2*M+1)*N-M**2-M),dt   
	REAL*8 G1,G2,BB1,B1,M1,F1,FF1,B,F


	A = 0.D0

	cont = 1
	DO i=1,N

		IF(i.EQ.N) THEN
			A(cont) = 1.D0
			cont = cont + M + 1
			GO TO 20
		END IF

		IF(i.LE.M) THEN
			jmin = 1
			jmax = i + M
		ELSE IF(i.GT.M.AND.i.LE.N-M) THEN
			jmin = i - M
			jmax = jmin + 2*M
		ELSE
			jmin = i - M
			jmax = N
		END IF

		IF(U(i).GE.0.D0) THEN
			G1 = U(i)
			G2 = 0.D0
		ELSE
			G2 = U(i)
			G1 = 0.D0
		END IF

		B = (dz(i) + dz(i-1))/dz(i)
		F = (dz(i+1) + dz(i+2))/dz(i+1)

		B1 = 0.D0
		BB1 = 0.D0

		IF(i.EQ.2) THEN
			B1 = -G1/dz(i)
		ELSE IF(i.GT.2) THEN
			B1 = -(B**2/(B**2-B))*G1/dz(i)
			BB1 = (1.D0/(B**2-B))*G1/dz(i)
		END IF

		F1 = 0.D0
		FF1 = 0.D0

		IF(i.EQ.N-1) THEN
			F1 = G2/dz(i+1)
		ELSE IF(i.LT.N-1) THEN
			F1 = (F**2/(F**2-F))*G2/dz(i+1)
			FF1 = -(1.D0/(F**2-F))*G2/dz(i+1)
		END IF

		M1 = 1/dt - B1 - BB1 - F1 - FF1

		DO j=jmin,jmax

			IF(j.EQ.i-2) THEN
				A(cont) = BB1
				GO TO 10
			ELSE IF(j.EQ.i-1) THEN
				A(cont) = B1
				GO TO 10
			ELSE IF(j.EQ.i) THEN
				A(cont) = M1
				GO TO 10
			ELSE IF(j.EQ.i+1) THEN
				A(cont) = F1
				GO TO 10
			ELSE
				A(cont) = FF1
			END IF

10			CONTINUE

			cont = cont + 1

		END DO

20		CONTINUE

	END DO


	RETURN
	END


************************************************************************
	SUBROUTINE CalcKS(Sin,S0,dt,K,N)
************************************************************************

	IMPLICIT NONE
	INTEGER i,N
	REAL*8 Sin,S0(N),dt,K(N)


	K(N) = Sin

	DO i=1,N-1
		K(i) = S0(i)/dt
	END DO


	RETURN
	END


************************************************************************
	SUBROUTINE Calcf(U,RHO,VISC,D,epsilon,f,N)
************************************************************************

	IMPLICIT NONE
	INTEGER i,N
	REAL*8 U(N),RHO(N),VISC,D,epsilon,f(N)
	REAL*8 Re,Yold,Y,ERRO


	DO i=1,N
		IF(ABS(U(i)).GT.1.D-3) THEN
			Re = RHO(i)*ABS(U(i))*D/VISC
		ELSE
			Re = RHO(i)*1.D-3*D/VISC
		END IF

		Y = -2.D0*DLOG10(epsilon/(3.7D0*D)+(7.D0/Re)**0.9D0)

		ERRO = 1.D0
		DO WHILE(ERRO.GT.1.D-6)
			Yold = Y
			Y = -2.D0*DLOG10(epsilon/(3.7D0*D)+2.51D0*Y/Re)
			ERRO = ABS(Y - Yold)
		END DO
		f = (1.D0/Y)**2.D0

		IF(f(i).GT.1.D3) f(i) = 1.D3

	END DO


	RETURN
	END


************************************************************************
	SUBROUTINE RESERVOIR(Gin,Tin,Pwf)
************************************************************************

	IMPLICIT NONE
	REAL*8 Gin,Tin,Pwf


	Gin = -50.D0
	Tin = 300.D0


	RETURN
	END