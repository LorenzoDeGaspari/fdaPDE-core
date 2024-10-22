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

#ifndef __INTEGRATORIGA_H__
#define __INTEGRATORIGA_H__

#include "../mesh/mesh.h"
#include "../../fields/scalar_expressions.h"
#include "../../utils/symbols.h"
#include "../../utils/integration/integrator_tables.h"
#include "../../utils/integration/integrator.h"

namespace fdapde {
namespace core {

// A set of utilities to perform numerical integration
// M: dimension of the domain of integration, R nurbs order, K number of quadrature nodes
template <int M, int R, int K = standard_iga_quadrature_rule<M, R>::K> class IntegratorIga {
   private:
    IntegratorTable<M, K, GaussLegendre> integration_table_;
   public:
    IntegratorIga() : integration_table_(IntegratorTable<M, K, GaussLegendre>()) {};

    // integrate a callable F over a mesh element e
    template <int N, typename F> double integrate(const ElementIga<M, N, R>& e, const F& f) const;
    // integrate a callable F over a triangualtion m
    template <int N, typename F> double integrate(const MeshIga<M, N, R>& m, const F& f) const;
    // computes \int_e [f * \phi] where \phi is a basis function over the *reference element*.
    template <int N, typename F, typename B, typename G>
    double integrate(const ElementIga<M, N, R>& e, const F& f, const B& phi, G& g) const;
    // integrate the weak form of operator L to produce its (i,j)-th discretization matrix element
    template <typename L, int N, typename F> double integrate(const ElementIga<M, N, R>& e, F& f) const;

    // getters
    template <int N> DMatrix<double> quadrature_nodes(const MeshIga<M, N, R>& m) const;
    std::size_t num_nodes() const { return integration_table_.num_nodes; }
};

// implementative details

// integration of bilinear form
template <int M, int R, int K>
template <typename L, int N, typename F>
double IntegratorIga<M, R, K>::integrate(const ElementIga<M, N, R>& e, F& f) const {
    // apply quadrature rule
    double value = 0;
    for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
        SVector<M> x = e.affine_map(integration_table_.nodes[iq]);
        if constexpr (std::remove_reference<L>::type::is_space_varying) {
            // space-varying case: forward the quadrature node index to non constant coefficients
            f.forward(integration_table_.num_nodes * e.ID() + iq);
        }
        value += f(x) * integration_table_.weights[iq];
    }
    // correct for measure of domain (element e)
    return value * e.integral_measure();
}

// perform integration of \int_e [f * \phi] using a basis system defined over the reference element and the change of
// variables formula: \int_e [f(x) * \phi(x)] = \int_{E} [f(J(X)) * \Phi(X)] |detJ|
// where J is the affine mapping from the reference element E to the physical element e
template <int M, int R, int K>
template <int N, typename F, typename B, typename G>
double IntegratorIga<M, R, K>::integrate(const ElementIga<M, N, R>& e, const F& f, const B& Phi, G & g) const {
    double value = 0;
    for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
        const SVector<M>& x = e.affine_map(integration_table_.nodes[iq]);
        if constexpr (std::is_base_of<ScalarExpr<N, F>, F>::value) {
            // functor f is evaluable at any point.
            SVector<N> Jx = e.parametrization()(x);   // map quadrature point on physical element e
            g.forward(integration_table_.num_nodes * e.ID() + iq);
            value += (f(Jx) * Phi(x) * g(x)) * integration_table_.weights[iq];
        } else {
            // as a fallback we assume f given as vector of values with the assumption that
            // f[integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature node.
            g.forward(integration_table_.num_nodes * e.ID() + iq);
            value += (f(integration_table_.num_nodes * e.ID() + iq, 0) * Phi(x) * g(x)) * integration_table_.weights[iq];
        }
    }
    // correct for measure of domain (element e)
    return value * e.integral_measure();
}

// integrate a callable F over a mesh element e. Do not require any particular structure for F
template <int M, int R, int K>
template <int N, typename F>
double IntegratorIga<M, R, K>::integrate(const ElementIga<M, N, R>& e, const F& f) const {
    double value = 0;
    for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
        if constexpr (std::is_invocable<F, SVector<N>>::value) {
            // functor f is evaluable at any point
            SVector<N> x = e.affine_map(integration_table_.nodes[iq]);   // map quadrature point onto e
            value += f(x) * integration_table_.weights[iq];
        } else {
            // as a fallback we assume f given as vector of values with the assumption that
            // f[integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature node.
            value += f(integration_table_.num_nodes * e.ID() + iq, 0) * integration_table_.weights[iq];
        }
    }
    // correct for measure of domain (element e)
    return value * e.integral_measure();
}

// integrate a callable F over the entire mesh m.
template <int M, int R, int K>
template <int N, typename F>
double IntegratorIga<M, R, K>::integrate(const MeshIga<M, N, R>& m, const F& f) const {
    double value = 0;
    // cycle over all mesh elements
    for (const auto& e : m) value += integrate(e, f);
    return value;
}

// returns all quadrature points on the mesh
template <int M, int R, int K>
template <int N>
DMatrix<double> IntegratorIga<M, R, K>::quadrature_nodes(const MeshIga<M, N, R>& m) const {
    DMatrix<double> quadrature_nodes;
    quadrature_nodes.resize(m.n_elements() * integration_table_.num_nodes, M);
    // cycle over all mesh elements
    for (const auto& e : m) {
        // for each quadrature node, map it onto the physical element e and store it
        for (size_t iq = 0; iq < integration_table_.num_nodes; ++iq) {
            quadrature_nodes.row(integration_table_.num_nodes * e.ID() + iq) = e.affine_map(integration_table_.nodes[iq]);
        }
    }
    return quadrature_nodes;
}

}   // namespace core
}   // namespace fdapde

#endif   // __INTEGRATORIGA_H__
