/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
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
 ***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "global.h"
#include "su3.h"
#include "gamma.h"
#include "start.h"
#include "linalg_eo.h"
#include "operator/tm_operators.h"
#include "boundary.h"
#include "gmres.h"
#include "solver.h"
#include "block.h"
#include "operator/Hopping_Matrix.h"
#include "solver_field.h"
#include "operator/D_psi.h"

void dummy_Di(spinor * const P, spinor * const Q, const int i) {
  Block_D_psi(&block_list[i], P, Q);
  return;
}


void Mtm_plus_block_psi(spinor * const l, spinor * const k, const int i) {
  block * blk = &block_list[i];
  int vol = (*blk).volume/2;
  Block_H_psi(blk, &g_spinor_field[DUM_MATRIX+1][i*vol], k, EO);
  mul_one_pm_imu_inv(&g_spinor_field[DUM_MATRIX+1][i*vol], +1., vol);
  Block_H_psi(blk, &g_spinor_field[DUM_MATRIX][i*vol], &g_spinor_field[DUM_MATRIX+1][i*vol], OE);
  mul_one_pm_imu_sub_mul(l, k, &g_spinor_field[DUM_MATRIX][i*vol], +1., vol);
  return;
}

void Mtm_plus_sym_block_psi(spinor * const l, spinor * const k, const int i) {
  block * blk = &block_list[i];
  int vol = (*blk).volume/2;
  Block_H_psi(blk, &g_spinor_field[DUM_MATRIX+1][i*vol], k, EO);
  mul_one_pm_imu_inv(&g_spinor_field[DUM_MATRIX+1][i*vol], +1., vol);
  Block_H_psi(blk, &g_spinor_field[DUM_MATRIX][i*vol], &g_spinor_field[DUM_MATRIX+1][i*vol], OE);
  mul_one_pm_imu_inv(&g_spinor_field[DUM_MATRIX][i*vol], +1., vol);
  diff(l, k, &g_spinor_field[DUM_MATRIX][i*vol], vol);
  return;
}


void dummy_D0(spinor * const P, spinor * const Q) {
  Block_D_psi(&block_list[0], P, Q);
  return;
}

void dummy_D1(spinor * const P, spinor * const Q) {
  Block_D_psi(&block_list[1], P, Q);
  return;
}

void Msap(spinor * const P, spinor * const Q, const int Ncy, const int Niter) {
  int blk, ncy = 0, eo, vol;
  spinor * r, * a, * b;
  double nrm;
  spinor ** solver_field = NULL;
  const int nr_sf = 3;

  /* 
   * here it would be probably better to get the working fields as a parameter 
   * from the calling function
   */
  init_solver_field(&solver_field, VOLUME, nr_sf);
  r = solver_field[0];
  a = solver_field[1];
  b = solver_field[2];

  for(ncy = 0; ncy < Ncy; ncy++) {
    /* compute the global residue        */
    /* this can be done more efficiently */
    /* here only a naive implementation  */
    for(eo = 0; eo < 2; eo++) {
      D_psi(r, P);
      diff(r, Q, r, VOLUME);
      nrm = square_norm(r, VOLUME, 1);
      if(g_proc_id == 0 && g_debug_level > 2 && eo == 1) {  /*  GG, was 1 */
	printf("Msap: %d %1.3e\n", ncy, nrm);
	fflush(stdout);
      }
      /* choose the even (odd) block */

      /*blk = eolist[eo];*/

      for (blk = 0; blk < nb_blocks; blk++) {
	if(block_list[blk].evenodd == eo) {
	  vol = block_list[blk].volume;

	  /* get part of r corresponding to block blk into b */
	  copy_global_to_block(b, r, blk);

	  mrblk(a, b, Niter, 1.e-31, 1, vol, &dummy_Di, blk);

	  /* add a up to full spinor P */
	  add_block_to_global(P, a, blk);
	}
      }
    }
  }
  finalize_solver(solver_field, nr_sf);
  return;
}

// This is a smoother based on the even/odd preconditioned CG
// it applies Ncy iterations of even/odd CG to spinor Q
// and stores the result in P

void CGeoSmoother(spinor * const P, spinor * const Q, const int Ncy, const int dummy) {
  spinor ** solver_field = NULL;
  const int nr_sf = 5;
  double musave = g_mu;
  g_mu = g_mu1;
  init_solver_field(&solver_field, VOLUMEPLUSRAND/2, nr_sf);

  convert_lexic_to_eo(solver_field[0], solver_field[1], Q);
  assign_mul_one_pm_imu_inv(solver_field[2], solver_field[0], +1., VOLUME/2);
    
  Hopping_Matrix(OE, solver_field[4], solver_field[2]); 
  /* The sign is plus, since in Hopping_Matrix */
  /* the minus is missing                      */
  assign_mul_add_r(solver_field[4], +1., solver_field[1], VOLUME/2);
  /* Do the inversion with the preconditioned  */
  /* matrix to get the odd sites               */
  gamma5(solver_field[4], solver_field[4], VOLUME/2);
  cg_her(solver_field[3], solver_field[4], Ncy, 1.e-8, 1, 
	 VOLUME/2, &Qtm_pm_psi);
  Qtm_minus_psi(solver_field[3], solver_field[3]);

  /* Reconstruct the even sites                */
  Hopping_Matrix(EO, solver_field[4], solver_field[3]);
  mul_one_pm_imu_inv(solver_field[4], +1., VOLUME/2);
  /* The sign is plus, since in Hopping_Matrix */
  /* the minus is missing                      */
  assign_add_mul_r(solver_field[2], solver_field[4], +1., VOLUME/2);

  convert_eo_to_lexic(P, solver_field[2], solver_field[3]); 
  g_mu = musave;
  finalize_solver(solver_field, nr_sf);
  return;  
}

void Msap_eo(spinor * const P, spinor * const Q, const int Ncy, const int Niter) {
  int blk, ncy = 0, eo, vol;
  spinor * r, * a, * b;
  double nrm;
  spinor * b_even, * b_odd, * a_even, * a_odd;
  spinor ** solver_field = NULL;
  const int nr_sf = 3;

  /* 
   * here it would be probably better to get the working fields as a parameter 
   * from the calling function
   */
  init_solver_field(&solver_field, VOLUME, nr_sf);
  r = solver_field[0];
  a = solver_field[1];
  b = solver_field[2];

  vol = block_list[0].volume/2;
  b_even = b;
  b_odd = b + vol + 1;
  a_even = a;
  a_odd = a + vol + 1;

  for(ncy = 0; ncy < Ncy; ncy++) {
    /* compute the global residue        */
    /* this can be done more efficiently */
    /* here only a naive implementation  */
    for(eo = 0; eo < 2; eo++) {
      D_psi(r, P);
      diff(r, Q, r, VOLUME);
      nrm = square_norm(r, VOLUME, 1);
      if(g_proc_id == 0 && g_debug_level > 2 && eo == 0) {
	printf("Msap_eo: %d %1.3e\n", ncy, nrm);
	fflush(stdout);
      }
      /* choose the even (odd) block */

      //#ifdef OMP
      //# pragma omp parallel for
      //#endif
      // OMP doesn't work right now because a_even, a_odd, b_even, b_odd are not threadsafe
      // also need to make sure that e.g. assign_mul_... and the linalg stuff do not 
      // start threads again...
      for (blk = 0; blk < nb_blocks; blk++) {
	if(block_list[blk].evenodd == eo) {
	  /* get part of r corresponding to block blk into b_even and b_odd */
	  copy_global_to_block_eo(b_even, b_odd, r, blk);

	  assign_mul_one_pm_imu_inv(a_even, b_even, +1., vol);
	  Block_H_psi(&block_list[blk], a_odd, a_even, OE);
	  /* a_odd = a_odd - b_odd */
	  diff(a_odd, b_odd, a_odd, vol);

	  mrblk(b_odd, a_odd, Niter, 1.e-31, 1, vol, &Mtm_plus_block_psi, blk);

	  Block_H_psi(&block_list[blk], b_even, b_odd, EO);
	  mul_one_pm_imu_inv(b_even, +1., vol);
	  /* a_even = a_even - b_even */
	  diff(a_even, a_even, b_even, vol);

	  /* add even and odd part up to full spinor P */
	  add_eo_block_to_global(P, a_even, b_odd, blk);
	}
      }
    }
  }
  finalize_solver(solver_field, nr_sf);
  return;
}
