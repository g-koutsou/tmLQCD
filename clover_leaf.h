/***********************************************************************
 * $Id: operator.c 1763 2011-04-21 11:51:47Z reker $ 
 *
 * Copyright (C) 2005 Martin Hasenbusch
 *               2011 Carsten Urbach
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

#ifndef _CLOVER_LEAF_H
#define _CLOVER_LEAF_H
#include "su3.h"

extern void sw_term();
extern double sw_trace(const int ieo);
extern void sw_invert(const int ieo);
extern void sw_deriv(const int ieo);
extern void sw_spinor(const int ieo, spinor * const kk, spinor * const ll);
extern void sw_all();

#endif