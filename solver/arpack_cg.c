/*****************************************************************************
 * Copyright (C) 2014 Abdou M. Abdel-Rehim
 *
 * Deflating CG using eigenvectors computed using ARPACK
 * eigenvectors used correspond to those with smallest magnitude
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef MPI
# include <mpi.h>
#endif

#include "global.h"
#include "gettime.h"
#include "linalg_eo.h"
#include "start.h"
#include "linalg/blas.h"
#include "linalg/lapack.h"
#include "solver_field.h"
#include "solver/arpack_cg.h"
#include "solver/cg_her.h"
#if (defined WRITE_EVS)
# include "io/spinor.h"
# include "read_input.h"
#endif

#include "operator.h"

/*  
    similar to scalar_prod() but for complex vectors instead of spinors 
*/
static _Complex double
vec_dot(int N, const _Complex double *S, const _Complex double *R)
{
  _Complex double ALIGN res = 0.0;
#ifdef MPI
  _Complex double ALIGN mres;
#endif

#ifdef OMP
#pragma omp parallel
  {
  int thread_num = omp_get_thread_num();
#endif

  _Complex double ALIGN ds,tr,ts,tt,ks,kc;
  const _Complex double *s,*r;

  ks = 0.0;
  kc = 0.0;

#if (defined BGL && defined XLC)
  __alignx(16, S);
  __alignx(16, R);
#endif

#ifdef OMP
#pragma omp for
#endif
  for (int ix = 0; ix < N; ix++)
  {
    s= S + ix;
    r= R + ix;
    
    ds = (*r) * conj(*s);

    /* Kahan Summation */
    tr=ds+kc;
    ts=tr+ks;
    tt=ts-ks;
    ks=ts;
    kc=tr-tt;
  }
  kc=ks+kc;

#ifdef OMP
  g_omp_acc_cp[thread_num] = kc;

  } /* OpenMP closing brace */

  /* having left the parallel section, we can now sum up the Kahan
     corrected sums from each thread into kc */
  for(int i = 0; i < omp_num_threads; ++i)
    res += g_omp_acc_cp[i];
#else
  res=kc;
#endif

#ifdef MPI
  MPI_Allreduce(&res, &mres, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  return(mres);
#endif
  return(res);
}

/*  
    similar to assign_add_mul() but for complex vectors instead of spinors 
*/
void
vec_axpy(const int N, const _Complex double c, _Complex double * const S, _Complex double * const R)
{
#ifdef OMP
#pragma omp parallel
  {
#endif
  _Complex double *r, *s;

#ifdef OMP
#pragma omp for
#endif
  for (int ix=0; ix<N; ix++)
  {
    r=(_Complex double *) R + ix;
    s=(_Complex double *) S + ix;

    (*r) += c * (*s);
  }
  
#ifdef OMP
  } /* OpenMP closing brace */
#endif

}


static void *
amalloc(size_t size)
{
  void *ptr;
#if (defined SSE || defined SSE2 || defined SSE3 || defined ALIGN)
  posix_memalign(&ptr, ALIGN_BASE+1, size);
  if(ptr==NULL)
    {
      if(g_proc_id == g_stdio_proc)
	fprintf(stderr,"posix_memalign() returned NULL in arpack_cg.\n");
      exit(1);
    }
#else
  ptr = malloc(size);
  if(ptr==NULL)
    {
      if(g_proc_id == g_stdio_proc)
	fprintf(stderr,"malloc() returned NULL in arpack_cg.\n");
      exit(1);
    }
#endif
  return ptr;
};

#define N_ARPACK_AUX_SPINORS 4
#define N_ARPACK_AUX_VECTORS 2

/* These are needed across CG calls, one for each operator, and their
   values need to remain unchanged once set */
static _Complex double *evecs_op[max_no_operators];
static _Complex double *evals_op[max_no_operators];
static _Complex double *H_op[max_no_operators];

/* These are needed across CG calls, one for each operator. Their
   values do not need to remain unchanged, but their sizes
   depend on the operator id */
static _Complex double *H_aux_op[max_no_operators];
static _Complex double *initwork_op[max_no_operators];

/* Number of rhs computed for each operator id. This is only important
   when ncurRHS == 0, so that we know we need to allocate memory and
   do the deflation, and when ncurRHS == nrhs-1, so that we know we
   should free the memory allocated */
static int ncurRHS_op[max_no_operators];

/* Number of converged eigenvectors for each operator id. Could be
   different for each operator. */
static int nconv_op[max_no_operators];

static int arpack_initialized = 0;

/* These could be allocated with each CG call, but
   we allocate once and reuse to save allocation overhead  */
static spinor *arpack_aux_spinors[N_ARPACK_AUX_SPINORS];
static _Complex double *arpack_aux_vectors[N_ARPACK_AUX_VECTORS];

static void
init_arpack_cg(int N)
{
  /* leading dimension for spinor vectors */
  size_t LDN;
  if(N==VOLUME)
    LDN = VOLUMEPLUSRAND;
  else
    LDN = VOLUMEPLUSRAND/2; 

  for(int i=0; i<N_ARPACK_AUX_SPINORS; i++)
    arpack_aux_spinors[i] = amalloc(LDN*sizeof(spinor));  

  for(int i=0; i<N_ARPACK_AUX_VECTORS; i++)
    arpack_aux_vectors[i] = amalloc(12*N*sizeof(_Complex double));

  arpack_initialized = 1;
  for(int i=0; i<max_no_operators; i++)
    ncurRHS_op[i] = 0;

  for(int i=0; i<max_no_operators; i++)
    nconv_op[i] = 0;
  return;
}

static void
finalize_arpack_cg(void)
{
  /* leading dimension for spinor vectors */
  for(int i=0; i<N_ARPACK_AUX_SPINORS; i++)
    free(arpack_aux_spinors[i]);

  for(int i=0; i<N_ARPACK_AUX_VECTORS; i++)
    free(arpack_aux_vectors[i]);

  arpack_initialized = 0;
  for(int i=0; i<max_no_operators; i++)
    ncurRHS_op[i] = 0;
  
  for(int i=0; i<max_no_operators; i++)
    nconv_op[i] = 0;

  return;
}

int arpack_cg(
     //solver params
     const int N,             /*(IN) Number of lattice sites for this process*/
     const int nrhs,          /*(IN) Number of right-hand sides to be solved*/ 
     const int nrhs1,         /*(IN) First number of right-hand sides to be solved using tolerance eps_sq1*/ 
     spinor * const x,        /*(IN/OUT) initial guess on input, solution on output for this RHS*/
     spinor * const b,        /*(IN) right-hand side*/
     matrix_mult f,           /*(IN) f(s,r) computes s=A*r, i.e. matrix-vector multiply*/
     const double eps_sq1,    /*(IN) squared tolerance of convergence of the linear system for systems 1 till nrhs1*/
     const double eps_sq,     /*(IN) squared tolerance of convergence of the linear system for systems nrhs1+1 till nrhs*/
     const double res_eps_sq, /*(IN) suqared tolerance for restarting cg */
     const int rel_prec,      /*(IN) 0 for using absoute error for convergence
                                     1 for using relative error for convergence*/
     const int maxit,         /*(IN) Maximum allowed number of iterations to solution for the linear system*/

     //parameters for arpack
     const int nev,                 /*(IN) number of eigenvectors to be computed by arpack*/
     const int ncv,                 /*(IN) size of the subspace used by arpack with the condition (nev+1) =< ncv*/
     double arpack_eig_tol,         /*(IN) tolerance for computing eigenvalues with arpack */
     int arpack_eig_maxiter,        /*(IN) maximum number of iterations to be used by arpack*/
     int kind,                      /*(IN) 0 for eigenvalues with smallest real part "SR"
                                           1 for eigenvalues with largest real part "LR"
                                           2 for eigenvalues with smallest absolute value "SM"
                                           3 for eigenvalues with largest absolute value "LM"
                                           4 for eigenvalues with smallest imaginary part "SI"
                                           5 for eigenvalues with largest imaginary part  "LI"*/
     int comp_evecs,                /*(IN) 0 don't compute the eiegnvalues and their residuals of the original system 
                                           1 compute the eigenvalues and the residuals for the original system (the orthonormal baiss
                                             still be used in deflation and they are not overwritten).*/
     int acc,                       /*(IN) 0 no polynomial acceleration
                                           1 use polynomial acceleration*/
     int cheb_k,                    /*(IN) degree of the Chebyshev polynomial (irrelevant if acc=0)*/
     double emin,                      /*(IN) lower end of the interval where the acceleration will be used (irrelevant if acc=0)*/
     double emax,                      /*(IN) upper end of the interval where the acceleration will be used (irrelevant if acc=0)*/
     char *arpack_logfile,           /*(IN) file name to be used for printing out debugging information from arpack*/
     const int op_id                 /* (IN) the operator ID. ARPACK will deflate for each new operator, and keep the
					eigenvectors alive separately for different operatos. Useful for interleaving up and down
					inversions */
	      )
{ 
  //Static variables and arrays.
  int parallel;        /* for parallel processing of the scalar products */
  #ifdef MPI
    parallel=1;
  #else
    parallel=0;
  #endif


  /*-------------------------------------------------------------
  //if this is the first right hand side, allocate memory, 
  //call arpack, and compute resiudals of eigenvectors if needed
  //-------------------------------------------------------------*/ 
  
  int ncurRHS = ncurRHS_op[op_id];
  _Complex double *evecs = evecs_op[op_id];
  _Complex double *evals = evals_op[op_id];
  _Complex double *H = H_op[op_id];
  _Complex double *H_aux = H_aux_op[op_id];
  _Complex double *initwork = initwork_op[op_id];
  if(!arpack_initialized)
    init_arpack_cg(N);

  spinor *ax = arpack_aux_spinors[0];
  spinor *r = arpack_aux_spinors[1];
  spinor *s0 = arpack_aux_spinors[2];
  spinor *s1 = arpack_aux_spinors[3];
  _Complex double *z0 = arpack_aux_vectors[0];
  _Complex double *z1 = arpack_aux_vectors[1];
  if(ncurRHS==0){
    evecs = amalloc(ncv*12*N*sizeof(_Complex double));
    evals = amalloc(ncv*sizeof(_Complex double));

    double et1 = gettime();
    int info_arpack = 0;
    int nconv = 0;
    evals_arpack(N, nev, ncv, kind, acc, cheb_k, emin, emax,
		 evals, evecs, arpack_eig_tol, arpack_eig_maxiter,
		 f, &info_arpack, &nconv, arpack_logfile);
    double et2 = gettime();

    if(info_arpack != 0){ //arpack didn't converge
      if(g_proc_id == g_stdio_proc)
        fprintf(stderr, "WARNING: ARPACK didn't converge. exiting..\n");
      return -1;
    }
    
    if(g_proc_id == g_stdio_proc)
    {
       fprintf(stdout, "ARPACK has computed %d eigenvectors\n", nconv);
       fprintf(stdout, "ARPACK time: %+e\n", et2 - et1);
    }
    nconv_op[op_id] = nconv;
    
    H = amalloc(nconv*nconv*sizeof(_Complex double)); 
    H_aux = amalloc(nconv*nconv*sizeof(_Complex double)); 
    initwork = amalloc(nconv*sizeof(_Complex double));

    //compute the elements of the hermitian matrix H 
    //leading dimension is nconv and active dimension is nconv
    for(int i=0; i<nconv; i++)
    {
      assign_complex_to_spinor(r, &evecs[i*12*N], 12*N);

       f(ax,r);
       double c1 = scalar_prod(r, ax, N, parallel);
       H[i+nconv*i] = creal(c1);  //diagonal should be real
       for(int j=i+1; j<nconv; j++)
       {
          assign_complex_to_spinor(r, &evecs[j*12*N], 12*N);
          c1 = scalar_prod(r, ax, N, parallel);
          H[j+nconv*i] = c1;
          H[i+nconv*j] = conj(c1); //enforce hermiticity
       }
     }

     //compute Ritz values and Ritz vectors if needed
    if( (nconv>0) && (comp_evecs !=0))
      {
	/* copy H into HU */
	int nconv_sq = nconv*nconv;
	int ONE = 1;
	_FT(zcopy)(&nconv_sq, H, &ONE, H_aux, &ONE);
	
	/* compute eigenvalues and eigenvectors of HU*/
	//SUBROUTINE ZHEEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK,INFO )
	int zheev_lwork = 3*nconv;
	_Complex double *zheev_work = amalloc(zheev_lwork*sizeof(_Complex double));
	double *zheev_rwork = amalloc(3*nconv*sizeof(double));
	double *hevals = amalloc(nconv*sizeof(double));
	int zheev_info;
	char cV='V', cU='U';
	_FT(zheev)(&cV, &cU, &nconv, H_aux, &nconv, hevals, zheev_work, &zheev_lwork, zheev_rwork, &zheev_info, 1, 1);
	
	if(zheev_info != 0)
	  {
	    if(g_proc_id == g_stdio_proc) 
	      {
	        fprintf(stderr,"Error in ZHEEV:, info =  %d\n",zheev_info); 
                fflush(stderr);
	      }
	    exit(1);
	  }
	free(zheev_rwork);
	free(zheev_work);
	
	//If you want to replace the schur (orthonormal) basis by eigen basis
	//use something like this. It is better to use the schur basis because
	//they are better conditioned. Use this part only to get the eigenvalues
	//and their resduals for the operator (D^\daggerD)
	//esize=(ncv-nconv)*12*N;
	//Zrestart_X(evecs,12*N,HU,12*N,nconv,nconv,&evecs[nconv*N],esize);
	
	/* compute residuals and print out results */
	
	if(g_proc_id == g_stdio_proc)
	  {fprintf(stdout,"Ritz values of A and their residulas (||A*x-lambda*x||/||x||\n"); 
	    fprintf(stdout,"=============================================================\n");
	    fflush(stdout);}
	
	for(int i=0; i<nconv; i++)
	  {
	    int Nx12=12*N;
	    char cN='N';
	    _Complex double zero = 0., zone = 1.;
	    int ONE = 1;
	    _FT(zgemv)(&cN, &Nx12, &nconv, &zone, evecs, &Nx12,
		       &H_aux[i*nconv], &ONE, &zero, z0, &ONE, 1);
	    
            assign_complex_to_spinor(r, z0, 12*N);	    
            double norm0 = square_norm(r, N, parallel);
            
            f(ax,r);
            mul_r(s0, hevals[i], r, N);
            diff(s1, ax, s0, N);
	    
	    double norm1 = square_norm(s1, N, parallel);
	    norm1 = sqrt(norm1/norm0);
	    
	    if(g_proc_id == g_stdio_proc)
	      {
		fprintf(stdout, "Eval[%06d]: %22.15E rnorm: %22.15E\n", i, hevals[i], norm1);
		fflush(stdout);
	      }
	  }
      
	free(hevals);
#ifdef WRITE_EVS
	 for(int i=0; i < nconv; i++)
	   {
	     int size = 12*N;
	     _Complex double zone = 1., zero = 0;
	     char cN = 'N';
	     int ONE = 1;
	     _FT(zgemv)(&cN, &size, &nconv, &zone, evecs, &size,
			&H_aux[(i+0)*nconv], &ONE, &zero, z0, &ONE, 1);
	     
	     assign_complex_to_spinor(s0, z0, size);
	     WRITER *writer = NULL;
	     int append = 0;
	     char fname[256];	
	     paramsPropagatorFormat *format = NULL;
	     int precision = 32;
	     int numb_flavs = 1;
	     
	     sprintf(fname, "ev.%04d.%05d", nstore, i);
	     construct_writer(&writer, fname, append);


	     format = construct_paramsPropagatorFormat(precision, numb_flavs);
	     write_propagator_format(writer, format);
	     free(format);	    

	     memset(z1, '\0', size*sizeof(_Complex double));
	     assign_complex_to_spinor(s1, z1,size);
	     
	     int status = write_spinor(writer, &s1, &s0, numb_flavs, precision);
	     destruct_writer(writer);
#if 1
	     sprintf(fname, "ev.%04d.%05d.txt", nstore, i);
	     FILE *fp = fopen(fname, "w");
	     for(int t=0; t<T; t++)
	       for(int z=0; z<LZ; z++)
		 for(int y=0; y<LY; y++)
		   for(int x=0; x<LX; x++) {		     
		     int j = x + y + z + t;
		     if(j%2 == 1) {
		       int k = z + LZ*(y + LY*(x + LX*t));
		       _Complex double *sp = &z0[(k/2)*12];
		       for(int cs=0; cs<12; cs++)
			 fprintf(fp,"%+e %+e\n", creal(sp[cs]), cimag(sp[cs]));
		     }
		   }
	     fclose(fp);

	     assign_complex_to_spinor(s0, z0, size);
	     f(s1, s0);
	     assign_spinor_to_complex(z0, s1, N);

	     assign_complex_to_spinor(s0, z1, size);
	     f(s1, s0);
	     assign_spinor_to_complex(z1, s1, N);

	     sprintf(fname, "Aev.%04d.%05d.txt", nstore, i);
	     fp = fopen(fname, "w");
	     for(int t=0; t<T; t++)
	       for(int z=0; z<LZ; z++)
		 for(int y=0; y<LY; y++)
		   for(int x=0; x<LX; x++) {		     
		     int j = x + y + z + t;
		     if(j%2 == 1) {
		       int k = z + LZ*(y + LY*(x + LX*t));
		       _Complex double *sp = &z0[(k/2)*12];
		       for(int cs=0; cs<12; cs++)
			 fprintf(fp,"%+e %+e\n", creal(sp[cs]), cimag(sp[cs]));
		     }
		   }
	     fclose(fp);
#endif /* 0 */	     
	   }
#endif /* def WRITE_EVS */
      }		//if( (nconv_arpack>0) && (comp_evecs !=0))

    evecs_op[op_id] = evecs;
    evals_op[op_id] = evals;
    H_op[op_id] = H;
    H_aux_op[op_id] = H_aux;
    initwork_op[op_id] = initwork;
  }		//if(ncurRHS==0)
  
  /* increment the RHS counter */
  ncurRHS = ncurRHS + 1; 
  int nconv = nconv_op[op_id];
  
  //set the tolerance to be used for this right-hand side 
  double eps_sq_used;
  if(ncurRHS > nrhs1){
    eps_sq_used = eps_sq;
  }
  else{
    eps_sq_used = eps_sq1;
  }
  
  if(g_proc_id == g_stdio_proc && g_debug_level > 0) {
    fprintf(stdout, "System %d, eps_sq %e\n", ncurRHS, eps_sq_used); 
    fflush(stdout);
  } 
  
  /*---------------------------------------------------------------*/
  /* Call init-CG until this right-hand side converges             */
  /*---------------------------------------------------------------*/

  int flag = -1;    	  /* System has not converged yet */
  int maxit_remain = maxit;
  int iters_tot = 0;
  double wall_time_ini = 0, wall_time_cg = 0;
  double restart_eps_sq_used = res_eps_sq;
  while( flag == -1 )
    {
      if(nconv > 0)
	{
	  /* --------------------------------------------------------- */
	  /* Perform init-CG with evecs vectors                        */
	  /* xinit = xinit + evecs*Hinv*evec'*(b-Ax0) 		   */
	  /* --------------------------------------------------------- */
	  double t0 = gettime();
	  
	  /*r0=b-Ax0*/
	  f(ax, x); /*ax = A*x */
	  diff(r, b, ax, N);  /* r=b-A*x */

#if 0 /* Naive way of getting guess. H is assumed dense, though it is practically diagonal */
	  /* x = x + evecs*inv(H)*evecs'*r */
	  for(int i=0; i < nconv; i++)
	    {
	      assign_complex_to_spinor(s0, &evecs[i*12*N], 12*N);
	      initwork[i] = scalar_prod(s0, r, N, parallel);
	    }
	  
	  /* solve the linear system H y = c */
	  int nconv_sq = nconv*nconv;
	  int ONE = 1;

	  
	  _FT(zcopy) (&nconv_sq, H, &ONE, H_aux, &ONE); /* copy H into H_aux */
	  
	  int IPIV[nconv];
	  int info_lapack;

#if 0
	  {
	    FILE *fp = fopen("H.dat","w");
	    for(int ic=0; ic<nconv; ic++) {
	      for(int jc=0; jc<nconv; jc++)
		fprintf(fp, "  %+16.12e%+16.12ej  ", creal(H[jc + ic*nconv]), cimag(H[jc + ic*nconv]));
	      fprintf(fp, "\n");
	    }
	    fclose(fp);
	  }

	  {
	    FILE *fp = fopen("b.dat","w");
	    for(int ic=0; ic<nconv; ic++)
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(initwork[ic]), cimag(initwork[ic]));	    
	    fclose(fp);
	  }
#endif
	  
	  _FT(zgesv) (&nconv, &ONE, H_aux, &nconv, IPIV, initwork, &nconv, &info_lapack);

#if 0
	  {
	    FILE *fp = fopen("H^{-1}b.dat","w");
	    for(int ic=0; ic<nconv; ic++)
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(initwork[ic]), cimag(initwork[ic]));	    
	    fclose(fp);
	  }
#endif
	  
	  if(info_lapack != 0)
	    {
	      if(g_proc_id == g_stdio_proc) {
		fprintf(stderr, "Error in ZGESV:, info =  %d\n", info_lapack); 
		fflush(stderr);
	      }
	      exit(1);
	    }
	  
	  /* x = x + evecs*inv(H)*evecs'*r */
	  for(int i=0; i<nconv; i++)
	    {
	      assign_complex_to_spinor(s0, &evecs[i*12*N], 12*N);
	      assign_add_mul(x, s0, initwork[i], N);
	    }
#endif /* End section with naive guess calculation */

#if 1 /* What follows assumes a diagonal H and so should be more optimal for computing the initial guess */

	  /* Initialize z1 to x0 */
	  assign_spinor_to_complex(z1, x, N);

	  /* Initialize z0 to r */
	  assign_spinor_to_complex(z0, r, N);

	  for(int i=0; i<nconv; i++) {
	    _Complex double udotr = vec_dot(12*N, &evecs[i*12*N], z0);
	    udotr = udotr / evals[i];
	    vec_axpy(12*N, udotr, &evecs[i*12*N], z1);
	  }
	    
	  /* Set x to accumulated guess */
	  assign_complex_to_spinor(x, z1, 12*N);
	  
#endif /* End section with optimized guess calculation */

#if 0
	  {
	    char *fname;
	    asprintf(&fname, "x%04d.dat", ncurRHS);
	    FILE *fp = fopen(fname,"w");
	    for(int i=0; i<N; i++) {
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s0.c0), cimag(x[i].s0.c0));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s0.c1), cimag(x[i].s0.c1));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s0.c2), cimag(x[i].s0.c2));
							           	              
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s1.c0), cimag(x[i].s1.c0));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s1.c1), cimag(x[i].s1.c1));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s1.c2), cimag(x[i].s1.c2));
							           	              
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s2.c0), cimag(x[i].s2.c0));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s2.c1), cimag(x[i].s2.c1));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s2.c2), cimag(x[i].s2.c2));
							           	              
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s3.c0), cimag(x[i].s3.c0));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s3.c1), cimag(x[i].s3.c1));
	      fprintf(fp, "  %+16.12e%+16.12ej\n", creal(x[i].s3.c2), cimag(x[i].s3.c2));
	    }
	    fclose(fp);
	  }
#endif
	  
	  /* compute elapsed time and add to accumulator */
	  t0 = gettime() - t0;
	  wall_time_ini += t0;
	}
      //which tolerance to use
      double tol_sq;
      if(eps_sq_used > restart_eps_sq_used)
	{
	  tol_sq = eps_sq_used;
	  flag = 1; //shouldn't restart again
	}
      else
	{
	  tol_sq = restart_eps_sq_used;
	}
      
      double t0 = gettime();
      int iters = cg_her(x, b, maxit_remain, tol_sq, rel_prec, N, f); 
      t0 = gettime() - t0;      
      wall_time_cg += t0;
      
      //check convergence
      if(iters == -1)
	{
	  //cg didn't converge
	  if(g_proc_id == g_stdio_proc) {
	    fprintf(stderr, "CG didn't converge within the maximum number of iterations in arpack_cg. Exiting...\n"); 
	    fflush(stderr);
	    exit(1);
	    
	  }
	} 
      else
	{
	  maxit_remain = maxit_remain - iters; //remaining number of iterations
	  restart_eps_sq_used = restart_eps_sq_used*res_eps_sq; //prepare for the next restart
	}
      iters_tot += iters;
    }
  /* end while (flag ==-1)               */
  
  /* ---------- */
  /* Reporting  */
  /* ---------- */
  /* compute the exact residual */
  f(ax, x);		/* ax= A*x */
  diff(r, b, ax, N);	/* r=b-A*x */	
  double normsq = square_norm(r, N, parallel);
  if(g_debug_level > 0 && g_proc_id == g_stdio_proc)
  {
    fprintf(stdout, "For this rhs:\n");
    fprintf(stdout, "Total initCG Wallclock : %+e\n", wall_time_ini);
    fprintf(stdout, "Total cg Wallclock : %+e\n", wall_time_cg);
    fprintf(stdout, "Iterations: %-d\n", iters_tot); 
    fprintf(stdout, "Actual Resid of LinSys  : %+e\n", normsq);
  }

  //free memory if this was your last system to solve for this operator id
  if(ncurRHS == nrhs)
    {
      free(evecs);
      free(evals);
      free(H);
      free(H_aux);
      free(initwork);
      ncurRHS = 0; 
      nconv = 0;
    }
  ncurRHS_op[op_id] = ncurRHS;
  nconv_op[op_id] = nconv;
  int arpack_not_done = 0;
  for(int i=0; i<max_no_operators; i++)
    {
      if(ncurRHS_op[i] != 0)
	arpack_not_done = 1;
    }
  if(!arpack_not_done)
    finalize_arpack_cg();
  
  return iters_tot;
}
 


      
