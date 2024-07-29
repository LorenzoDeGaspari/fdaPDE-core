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

#ifndef __IGA_DIFFUSION_H__
#define __IGA_DIFFUSION_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../fields/matrix_field.h"
#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../iga_symbols.h"

namespace fdapde {
namespace core {

// diffusion operator -div(K*surface_grad(.)) (anisotropic and non-stationary diffusion)
template <typename T> class Diffusion<IGA, T> : public DifferentialExpr<Diffusion<IGA, T>> {
    // perform compile-time sanity checks
    static_assert(
      std::is_base_of<MatrixBase, T>::value ||            // space-varying case
      std::is_base_of<Eigen::MatrixBase<T>, T>::value);   // constant coefficient case
   private:
    T K_;   // diffusion tensor (either constant or space-varying)
   public:
    enum {
        is_space_varying = true,
        is_symmetric = true
    };

    // constructor
    Diffusion() = default;
    explicit Diffusion(const T& K) : K_(K) { }
    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_IGA_MEM_BUFFER_SYMBOLS(mem_buffer);
	// non unitary or anisotropic diffusion: (\Nabla nurb_i).dot(InvG*\Nabla \nurb_j)*\g
	return -(nabla_nurb_i).dot(K_ * (InvG * nabla_nurb_j))*g;
    }
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __IGA_DIFFUSION_H__
