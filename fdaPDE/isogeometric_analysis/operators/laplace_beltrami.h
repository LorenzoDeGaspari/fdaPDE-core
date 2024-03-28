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

#ifndef __IGA_LAPLACEBELTRAMI_H__
#define __IGA_LAPLACEBELTRAMI_H__

#include <type_traits>

#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../iga_symbols.h"

namespace fdapde {
namespace core {

// laplacian operator (isotropic and stationary diffusion)
template <> struct LaplaceBeltrami<IGA> : public DifferentialExpr<LaplaceBeltrami<IGA>> {
    enum {
        is_space_varying = false,
        is_symmetric = true
    };

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_IGA_MEM_BUFFER_SYMBOLS(mem_buffer);
        // isotropic unitary diffusion: -(\Nabla nurb_i).dot(InvG*\Nabla nurb_j)*\g
        return -(nabla_nurb_i).dot(InvG * nabla_nurb_j)*g;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __IGA_LAPLACEBELTRAMI_H__
