// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_ISOGEOMETRIC_ANALYSIS_MODULE_H__
#define __FDAPDE_ISOGEOMETRIC_ANALYSIS_MODULE_H__

#include "isogeometric_analysis/basis/nurbs.h"
#include "isogeometric_analysis/basis/nurbs_basis.h"
#include "isogeometric_analysis/mesh/mesh.h"
#include "isogeometric_analysis/mesh/parametrization.h"
#include "isogeometric_analysis/integration/integrator.h"
#include "isogeometric_analysis/operators/advection.h"
#include "isogeometric_analysis/operators/reaction.h"
#include "isogeometric_analysis/operators/laplace_beltrami.h"
#include "isogeometric_analysis/operators/diffusion.h"
#include "isogeometric_analysis/iga_assembler.h"
#include "isogeometric_analysis/solvers/iga_linear_elliptic_solver.h"
#include "isogeometric_analysis/solvers/iga_solver_selector.h"

#endif // __FDAPDE_ISOGEOMETRIC_ANALYSIS_MODULE_H__
