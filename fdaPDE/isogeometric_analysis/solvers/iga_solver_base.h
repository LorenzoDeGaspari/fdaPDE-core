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

#ifndef __IGA_SOLVER_BASE_H__
#define __IGA_SOLVER_BASE_H__

#include <exception>

#include "../integration/integrator.h"
#include "../../utils/symbols.h"
#include "../../utils/traits.h"
#include "../../utils/combinatorics.h"
#include "../basis/nurbs_basis.h"
#include "../iga_assembler.h"
#include "../iga_symbols.h"
#include "../operators/reaction.h"   // for mass-matrix computation

namespace fdapde {
namespace core {

// forward declaration
template <typename PDE> struct is_pde;
  
// base class for the definition of a general solver based on the Finite Element Method
template <typename D, typename E, typename F, typename... Ts> class IGASolverBase {
   public:
    typedef std::tuple<Ts...> SolverArgs;
    enum {
        iga_order = std::tuple_element <0, SolverArgs>::type::value,
    };

    typedef D DomainType;
    typedef IntegratorIga<DomainType::local_dimension, iga_order, 4> QuadratureRule;
    typedef NurbsBasis<D::local_dimension,iga_order> FunctionBasis;
    typedef FunctionBasis FunctionSpace;

    // constructor
    IGASolverBase() = default;

    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& R1() const { return R1_; }
    const SpMatrix<double>& R0() const { return R0_; }
    const QuadratureRule& integrator() const { return integrator_; }
    const FunctionBasis& basis() const { return nurbs_basis_; }
    std::size_t n_dofs() const { return n_dofs_; }   // number of degrees of freedom (FEM linear system's unknowns)
    DMatrix<double> dofs_coords(const DomainType& mesh);

    // flags
    bool is_init = false;   // notified true if initialization occurred with no errors
    bool success = false;   // notified true if problem solved with no errors

    template <typename PDE> void init(const PDE& pde);
    template <typename PDE> void set_dirichlet_bc(const PDE& pde);

    struct boundary_dofs_iterator {   // range-for loop over boundary dofs
       private:
        friend IGASolverBase;
        const IGASolverBase* iga_solver_;
        int index_;   // current boundary dof
        boundary_dofs_iterator(const IGASolverBase* iga_solver, int index) : iga_solver_(iga_solver), index_(index) {};
       public:
        // fetch next boundary dof
        boundary_dofs_iterator& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < iga_solver_->n_dofs_ && iga_solver_->boundary_dofs_(index_,0) == 0; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_dofs_iterator& lhs, const boundary_dofs_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    boundary_dofs_iterator boundary_dofs_begin() const { return boundary_dofs_iterator(this, 0); }
    boundary_dofs_iterator boundary_dofs_end() const { return boundary_dofs_iterator(this, n_dofs_); }
  
   protected:
    QuadratureRule integrator_ {};       // default to a quadrature rule which is exact for the considered FEM order
    FunctionBasis nurbs_basis_ {};          // basis over the whole domain
    DMatrix<double> solution_;           // vector of coefficients of the approximate solution
    DMatrix<double> force_;              // discretized force [u]_i = \int_D f*\psi_i
    SpMatrix<double> R1_;   // [R1_]_{ij} = a(\psi_i, \psi_j), being a(.,.) the bilinear form of the problem
    SpMatrix<double> R0_;   // mass matrix, [R0_]_{ij} = \int_D (\psi_i * \psi_j)

    std::size_t n_dofs_ = 0;        // degrees of freedom, i.e. the maximum ID in the dof_table_
    DMatrix<std::size_t> boundary_dofs_;    // unknowns on the boundary of the domain, for boundary conditions prescription
   
   private:
    void enumerate_dofs(const DomainType& mesh);

};

// implementative details

// initialize solver
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void IGASolverBase<D, E, F, Ts...>::init(const PDE& pde) {
    static_assert(is_pde<PDE>::value, "not a valid PDE");
    nurbs_basis_ = pde.domain().basis();
    enumerate_dofs(pde.domain());
    // assemble discretization matrix for given operator    
    Assembler<IGA, D, FunctionBasis, QuadratureRule> assembler(pde.domain(), integrator_, n_dofs_, nurbs_basis_);
    R1_ = assembler.discretize_operator(pde.differential_operator());
    R1_.makeCompressed();
    // assemble forcing vector
    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m;             // number of time points
    if constexpr (!std::is_base_of<ScalarBase, F>::value) {
        m = pde.forcing_data().cols();
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(0));

        // iterate over time steps if a space-time PDE is supplied
        if constexpr (is_parabolic<E>::value) {
            for (std::size_t i = 1; i < m; ++i) {
                force_.block(n * i, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(i));
            }
        }
    } else {
        // TODO: support space-time callable forcing for parabolic problems
        m = 1;
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data());
    }
    // compute mass matrix [R0]_{ij} = \int_{\Omega} \phi_i \phi_j
    R0_ = assembler.discretize_operator(Reaction<IGA, double>(1.0));
    is_init = true;
    return;
}

// impose dirichlet boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void IGASolverBase<D, E, F, Ts...>::set_dirichlet_bc(const PDE& pde) {
    static_assert(is_pde<PDE>::value, "not a valid PDE");
    if (!is_init) throw std::runtime_error("solver must be initialized first!");
    for (auto it = boundary_dofs_begin(); it != boundary_dofs_end(); ++it) {
      R1_.row(*it) *= 0;            // zero all entries of this row
      R1_.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
	
      // TODO: currently only homogeneous case supported
      force_.coeffRef(*it, 0) = 0;   // impose boundary value on forcing term
    }
    return;
}
  
// builds a node enumeration for the support of a basis of order R. Specialization for 2D domains
template <typename D, typename E, typename F, typename... Ts>
void IGASolverBase<D, E, F, Ts...>::enumerate_dofs(const D& mesh) {
  if(n_dofs_ != 0) return; // return early if dofs already computed
    n_dofs_ = mesh.basis().size();
    boundary_dofs_ = mesh.boundary_dofs();
}

template <typename D, typename E, typename F, typename... Ts>
DMatrix<double> IGASolverBase<D, E, F, Ts...>::dofs_coords(const D& mesh){
    DMatrix<double> coords;
    return coords;
}
  
}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_BASE_H__
