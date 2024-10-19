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

#ifndef __IGA_ASSEMBLER_H__
#define __IGA_ASSEMBLER_H__

#include <memory>

#include "../fields/field_ptrs.h"
#include "../fields/scalar_field.h"
#include "../fields/vector_field.h"
#include "../mesh/mesh.h"
#include "../pde/assembler.h"
#include "../utils/compile_time.h"
#include "../utils/integration/integrator.h"
#include "../utils/symbols.h"
#include "basis/nurbs_basis.h"
#include "mesh/mesh.h"
#include "iga_symbols.h"


namespace fdapde {
namespace core {

// isogeometric method assembler
template <typename D, typename B, typename I> class Assembler<IGA, D, B, I> {
   private:
    std::size_t n_basis = n_basis;
    const D& mesh_;          // problem domain
    const I& integrator_;    // quadrature rule
    B reference_basis_;   // functional basis 
    int dof_;                // overall number of unknowns in IGA linear system
    DVector<double> f_;   // for non-linear operators, the estimate of the approximated solution 
   public:
    Assembler(const D& mesh, const I& integrator, int n_dofs, const B& ref_basis ) : mesh_(mesh), integrator_(integrator),
    dof_(n_dofs), reference_basis_(ref_basis) { n_basis = reference_basis_.size();};
    Assembler(const D& mesh, const I& integrator, int n_dofs,const B& ref_basis, const DVector<double>& f) : mesh_(mesh), integrator_(integrator), 
    dof_(n_dofs), reference_basis_(ref_basis), f_(f) {n_basis = reference_basis_.size();};
    
    // discretization methods
    template <typename E> SpMatrix<double> discretize_operator(const E& op);
    template <typename F> DVector<double> discretize_forcing(const F& force);
};

// implementative details

// assembly for the discretization matrix of a general operator L
template <typename D, typename B, typename I>
template <typename E>
SpMatrix<double> Assembler<IGA, D, B, I>::discretize_operator(const E& op) {
    constexpr std::size_t M = D::local_dimension;
    constexpr std::size_t N = D::embedding_dimension;
    constexpr std::size_t R = B::order;
    std::vector<Eigen::Triplet<double>> triplet_list;   // store triplets (node_i, node_j, integral_value)
    SpMatrix<double> discretization_matrix;
    // properly preallocate memory to avoid reallocations
    triplet_list.reserve(mesh_.n_elements() * std::pow(R+1,M));
    discretization_matrix.resize(dof_, dof_);

    // prepare space for bilinear form components
    using BasisType = typename B::ElementType;
    using NablaType = decltype(std::declval<BasisType>().derive());
    using JacobianType = typename D::DerivativeType;

    BasisType buff_nurb_i, buff_nurb_j;               // basis functions \nurb_i, \nurb_j
    NablaType buff_nabla_nurb_i, buff_nabla_nurb_j;   // gradient of basis functions \nabla \nurb_i, \nabla \nurb_j
    
    JacobianType F;
    F = mesh_.gradient();                // gradient of parametrization

    // InvG
    DMatrix<double> qn = integrator_.quadrature_nodes(mesh_);
    DMatrix<double, Eigen::RowMajor> invg_data;
    DMatrix<double, Eigen::RowMajor> g_data;
    invg_data.resize(integrator_.num_nodes() * mesh_.n_elements() * M * M, 1);
    g_data.resize(integrator_.num_nodes() * mesh_.n_elements(), 1);
    for (std::size_t i = 0; i < qn.rows(); ++i) {
        auto tmp = F(qn.row(i));
        auto tmp1 = tmp.transpose() * tmp;
        auto tmp2 = tmp1.inverse();
        auto tmp3 = std::sqrt(tmp1.determinant());
        for(std::size_t j = 0; j < M; ++j){
            for(std::size_t k = 0; k < M; ++k){
                invg_data( (i * M + j) * M + k) = tmp2(j,k);
            }
        }
        g_data(i) = tmp3;
    }
    MatrixDataWrapper<M,M,M> InvG(invg_data);
    ScalarDataWrapper<M> g(g_data);

    DVector<double> f;

    // prepare buffer to be sent to bilinear form
    auto mem_buffer = std::make_tuple(
      ScalarPtr(&buff_nurb_i), ScalarPtr(&buff_nurb_j), VectorPtr(&buff_nabla_nurb_i), VectorPtr(&buff_nabla_nurb_j),
      MatrixPtr(&F), MatrixPtr(&InvG), ScalarPtr(&g), &f); 

    
    // develop bilinear form expression in an integrable field here once
    auto weak_form = op.integrate(mem_buffer);   // let the compiler deduce the type of the expression template!

    std::size_t current_id;
    
    // cycle over all mesh elements
    for (const auto& e : mesh_) {
      // update elements related informations
      current_id = e.ID(); // element ID
      
      if(!is_empty(f_)) // should be bypassed in case of linear operators via an if constexpr!!!
	    for(std::size_t dof = 0; dof < n_basis; dof++) { f[dof] = f_[dof]; } 

        // consider all pair of nodes
        for (std::size_t i = 0; i < e.n_functions(); ++i) {
            buff_nurb_i = reference_basis_[e[i]];
            buff_nabla_nurb_i = buff_nurb_i.derive();   // update buffers content
            for (std::size_t j = 0; j < e.n_functions(); ++j) {
                buff_nurb_j = reference_basis_[e[j]];
                buff_nabla_nurb_j = buff_nurb_j.derive();   // update buffers content
                if constexpr (is_symmetric<decltype(op)>::value) {
                    // compute only half of the discretization matrix if the operator is symmetric
                    if (e[i] >= e[j]) {
                        double value = integrator_.template integrate<decltype(op)>(e, weak_form);

			// linearity of the integral is implicitly used during matrix construction, since duplicated
                        // triplets are summed up, see Eigen docs for more details
                        triplet_list.emplace_back(e[i], e[j], value);
                    }
                } else {
                    // not any optimization to perform in the general case
                    double value = integrator_.template integrate<decltype(op)>(e, weak_form);
                    triplet_list.emplace_back(e[i], e[j], value);
                }
            }
        }
    }
    
    // matrix assembled
    discretization_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    discretization_matrix.makeCompressed();
    
    // return just half of the discretization matrix if the form is symmetric (lower triangular part)
    if constexpr (is_symmetric<decltype(op)>::value)
        return discretization_matrix.selfadjointView<Eigen::Lower>();
    else
        return discretization_matrix;
        
};

template <typename D, typename B, typename I>
template <typename F>
DVector<double> Assembler<IGA, D, B, I>::discretize_forcing(const F& f) {
    constexpr std::size_t M = D::local_dimension;
    constexpr std::size_t N = D::embedding_dimension;
    constexpr std::size_t R = B::order;
    using JacobianType = typename D::DerivativeType;
    JacobianType jac;
    jac = mesh_.gradient();                // gradient of parametrization

    DMatrix<double> qn = integrator_.quadrature_nodes(mesh_);
    DMatrix<double, Eigen::RowMajor> g_data;
    g_data.resize(integrator_.num_nodes() * mesh_.n_elements(), 1);
    for (std::size_t i = 0; i < qn.rows(); ++i) {
        auto tmp = jac(qn.row(i));
        auto tmp1 = tmp.transpose() * tmp;
        auto tmp2 = std::sqrt(tmp1.determinant());
        g_data(i) = tmp2;
    }
    ScalarDataWrapper<M> g(g_data);

    // allocate space for result vector
    DVector<double> discretization_vector {};
    discretization_vector.resize(dof_, 1);   // there are as many basis functions as degrees of freedom on the mesh
    discretization_vector.fill(0);           // init result vector to zero

    // build forcing vector
    for (const auto& e : mesh_) {
        for (size_t i = 0; i < e.n_functions(); ++i) {
            // integrate \int_e [f*\nurb], exploit integral linearity
            discretization_vector[e[i]] += integrator_.integrate(e, f, reference_basis_[e[i]], g);
        }
    }
    return discretization_vector;
}

}   // namespace core
}   // namespace fdapde

#endif   // __IGA_ASSEMBLER_H__
