/************************************************************************* 
 * Interface for solving the eigenvalue problem A*x=lambda*x for a complex 
 * matrix A using ARPACK. The matrix is accessed through a matrix-vector 
 * multiplication function.
 *
 * Author: A.M. Abdel-Rehim, 2014
 *
 * For reference see the driver programs zndrv1 in the EXAMPLES 
 * subdriectories of ARPACK and PARPACK and the documentation.
 *
 * This file is part of tmLQCD software suite
 ***********************************************************************/

#ifndef _EIGENVALUES_ARPACK
#define _EIGENVALUES_ARPACK

#include "su3.h"
#include "solver/matrix_mult_typedef.h"
#include "linalg/arpack.h"
#include "memalloc.h"
#include "solver/precon.h"
#include "io/eospinor.ih"


/*compute nev eigenvectors using ARPACK and PARPACK*/
void evals_arpack(
  int n, 
  int nev, 
  int ncv, 
  int which,
  int use_acc,
  int cheb_k,
  double amin,
  double amax,
  _Complex double *evals, 
  _Complex double *v,
  double tol, 
  int maxiter, 
  matrix_mult av, 
  int *info, 
  int *nconv,
  char *arpack_logfile);
/*
  compute nev eigenvectors using ARPACK and PARPACK
  n     : (IN) size of the local lattice
  nev   : (IN) number of eigenvectors requested.
  ncv   : (IN) size of the subspace used to compute eigenvectors (nev+1) =< ncv < 12*n
          where 12n is the size of the matrix under consideration
  which : (IN) which eigenvectors to compute. Choices are:
          0: smallest real part "SR"
          1: largest real part "LR"
          2: smallest absolute value "SM"
          3: largest absolute value "LM"
          4: smallest imaginary part "SI"
          5: largest imaginary part "LI"
  use_acc: (IN) specify the polynomial acceleration mode
                0 no acceleration
                1 use acceleration by computing the eigenvectors of a shifted-normalized chebyshev polynomial
  cheb_k   : (IN) degree of the chebyshev polynomial to be used for acceleration (irrelevant when use_acc=0 and init_resid_arpack=0)
  amin,amax: (IN) bounds of the interval [amin,amax] for the acceleration polynomial (irrelevant when use_acc=0 and init_resid_arpack=0)
  store_basis: (IN) option to store basis vectors to disk such that they can be read later
                    0 don't store basis vectors
                    1 store basis vectors
  basis_fname: (IN) file name used to store the basis vectors
                    file names will be of the format
                    basis_fname.xxxxx where xxxxx will be the basis vector number with leading zeros
  basis_prec : (IN) precision used to write the basis vectors
                    0 single precision
                    1 double precision
  evals : (OUT) array of size nev+1 which has the computed nev Ritz values
  v     : computed eigenvectors. Size is n*ncv spinors.
  tol    : Requested tolerance for the accuracy of the computed eigenvectors.
           A value of 0 means machine precision.
  maxiter: maximum number of restarts (iterations) allowed to be used by ARPACK
  av     : operator for computing the action of the matrix on the vector
           av(vout,vin) where vout is output spinor and vin is input spinors.
  info   : output from arpack. 0 means that it converged to the desired tolerance. 
           otherwise, an error message is printed to stderr 
  nconv  : actual number of converged eigenvectors.
  arpack_logfile: name for the logfile to be used by arpack
*/ 


#endif
