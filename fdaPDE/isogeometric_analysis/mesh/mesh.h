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

#ifndef __MESHIGA_H__
#define __MESHIGA_H__

#include <Eigen/Core>
#include <array>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../../utils/combinatorics.h"
#include "../../utils/symbols.h"
#include "../basis/nurbs_basis.h"
#include "../../fields.h"
#include "parametrization.h"

namespace fdapde{
namespace core{

template <int M, int N, int R> class ElementIga {
    
    private:

        DVector<std::size_t> functions_; // indexes of the basis functions that have support in the element
        std::size_t ID_; // id of the element
        std::shared_ptr<MeshParametrization<M,N,R>> parametrization_;
        std::shared_ptr<ParametrizationDerivative<M,N,R>> gradient_;

    public:

        ElementIga() = default;
        ElementIga(const DVector<std::size_t> & functions, std::size_t ID) : functions_(functions), ID_(ID) {}
        std::size_t ID() const { return ID_; }
        const DVector<std::size_t> & functions() const { return functions_; }
        std::size_t n_functions() const { return functions_.rows(); }
        // overload operator[] to get directly the i-th index
        std::size_t operator[](std::size_t i) { return functions_[i]; }

};

// M local space dimension, N embedding space dimension
template <int M, int N, int R> class MeshIga{

    protected:

        SVector<M,DVector<double>> knots_;
        Tensor<double,M> weights_; // tensor of weights
        Tensor<double,M+1> control_points_; // tensor of the control points (for each weight there is a N-dimensional control point)
        VectorField<M, N, MeshParametrization<N,M,R>> parametrization_;
        MatrixField<M,N,M,ParametrizationDerivative<M,N,R>> gradient_;

        DMatrix<double> nodes_ {};
        DMatrix<std::size_t> boundary_ {};
        DMatrix<std::size_t, Eigen::RowMajor> elements_ {};
        DMatrix<std::size_t> neighbors_ {};
        std::vector<ElementIga<M,N,R>> elements_cache_;

    public:

        MeshIga() = default;
        MeshIga(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M+1>& control_points);

        VectorField<M, N, MeshParametrization<N,M,R>> parametrization() const { return parametrization_; }
        MatrixField<M,N,M,ParametrizationDerivative<M,N,R>> gradient() const { return gradient_; };

        const ElementIga<M,N,R> & element(std::size_t ID) const { return elements_cache_[ID]; }
        ElementIga<M,N,R>& element(std::size_t ID) { return elements_cache_[ID]; }
        SVector<N> node(int ID) const { return nodes_.row(ID); }
        bool is_on_boundary(int ID) const { return boundary_(ID) == 1; }
        const DMatrix<double>& nodes() const { return nodes_; }
        const DMatrix<std::size_t, Eigen::RowMajor>& elements() const { return elements_; }
        const DMatrix<std::size_t> & neighbors() const { return neighbors_; }
        const DMatrix<std::size_t>& boundary() const { return boundary_; }
        int n_elements() const { return elements_.rows(); }
        int n_nodes() const { return nodes_.rows(); }

        struct iterator {   // range-for loop over mesh elements
            private:
                friend MeshIga;
                const MeshIga* mesh_;
                int index_;   // current element
                iterator(const MeshIga* mesh, int index) : mesh_(mesh), index_(index) {};
            public:
                // increment current iteration index and return this iterator
                iterator& operator++() {
                    ++index_;
                    return *this;
                }
                const ElementIga<M,N,R>& operator*() { return mesh_->element(index_); }
                friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
                const ElementIga<M,N,R>& operator*() const { return mesh_->element(index_); }
        };

        struct boundary_iterator {   // range-for loop over boundary nodes
            private:
                friend MeshIga;
                const MeshIga* mesh_;
                int index_;   // current boundary node
                boundary_iterator(const MeshIga* mesh, int index) : mesh_(mesh), index_(index) {};
            public:
                // fetch next boundary node
                boundary_iterator& operator++() {
                    index_++;
                    // scan until all nodes have been visited or a boundary node is not found
                    for (; index_ < mesh_->n_nodes() && mesh_->is_on_boundary(index_) != true; ++index_)
                        ;
                    return *this;
                }
                int operator*() const { return index_; }
                friend bool operator!=(const boundary_iterator& lhs, const boundary_iterator& rhs) {
                    return lhs.index_ != rhs.index_;
                }
        };

        iterator begin() const { return iterator(this, 0); }
        iterator end() const { return iterator(this, elements_.rows()); }
        boundary_iterator boundary_begin() const { return boundary_iterator(this, 0); }
        boundary_iterator boundary_end() const { return boundary_iterator(this, n_nodes()); }

};

template <int M, int N, int R>
MeshIga<M,N,R>::MeshIga(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M+1>& control_points) :
    knots_(knots), weights_(weights), control_points_(control_points) {

    // build the domain parametrization function and its derivative
    std::vector<MeshParametrization<M,N,R>> param;
    std::array<std::array<ParametrizationDerivative<M,N,R>,M>,N> grad;
    param.reserve(N);
    // define i-th element of parametrization
    // define i,j-th element of the gradient of param (=partial derivative of param_i w.r.t. x_j)
    for (std::size_t i = 0; i < N; ++i) {
        param.emplace_back(knots,weights,control_points,i);
        for(std::size_t j=0;j<M;++j){
            grad[i][j]=ParametrizationDerivative<M,N,R>(knots,weights,control_points,i,j);
        }
    }
    // wrap the parametrization components into a vectorfield
    parametrization_ = VectorField<M, N, MeshParametrization<N,M,R>>(param);
    // wrap the gradient components into a matrixfield
    gradient_ = MatrixField<M,N,M,ParametrizationDerivative<M,N,R>>(grad);

}

}; // namespace core
}; // namespace fdapde
  
#endif   // __MESHIGA_H__
