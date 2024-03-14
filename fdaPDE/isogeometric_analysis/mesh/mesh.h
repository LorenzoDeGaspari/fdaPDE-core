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

        using ParametrizationType = MeshParametrization<M,N,R>;
        using GradientType = ParametrizationDerivative<M,N,R>;

        DVector<std::size_t> functions_; // indexes of the basis functions that have support in the element
        std::size_t ID_; // id of the element
        std::shared_ptr<ParametrizationType> parametrization_;
        std::shared_ptr<GradientType> gradient_;

    public:

        ElementIga() = default;
        ElementIga(const DVector<std::size_t> & functions, std::size_t ID, 
        const ParametrizationType & parametrization, const GradientType & gradient) 
        : functions_(functions), ID_(ID), parametrization_(std::make_shared<ParametrizationType>(parametrization)),
        gradient_(std::make_shared<GradientType>(gradient))  {}
        const VectorField<M, N, ParametrizationType>& parametrization() const { return *parametrization_; }
        const MatrixField<M,N,M,GradientType>& gradient() const { return *gradient_; };

        std::size_t ID() const { return ID_; }
        const DVector<std::size_t> & functions() const { return functions_; }
        std::size_t n_functions() const { return functions_.rows(); }
        // overload operator[] to get directly the i-th index
        std::size_t operator[](std::size_t i) { return functions_[i]; }

};

// N local space dimension, M embedding space dimension
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

        const VectorField<M, N, MeshParametrization<N,M,R>>& parametrization() const { return parametrization_; }
        const MatrixField<M,N,M,ParametrizationDerivative<M,N,R>>& gradient() const { return gradient_; }

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

    // setting the dimensions of nodes, boundary and elements matrices
    std::size_t rows = 1;
    std::size_t element_rows = 1;
    std::size_t temp=1;
    SVector<M,std::size_t> strides;
    for (std::size_t i=0; i<M; ++i){
        rows*=knots[i].size();
        element_rows*=(knots[i].size()-1);
        strides[i]=temp;
        temp*=knots[i].size();
    }
    nodes_.resize(rows,M);
    boundary_.resize(rows,1);
    elements_.resize(element_rows,(1<<M)-1);

    // filling the nodes matrix with the cartesian product of knots
    // filling column by column
    // each column is formed by repeating over and over the i-th knot vector
    // where each of its elements is repeated *stride* times (changes at each cycle)
    // this ensures that all the possible tuples of knots are considered
    // boundary points are the ones in which at least one component is the first or the last point of a knot vector
    for(std::size_t i=0; i<M; ++i){ // cycle over columns
        for(std::size_t j=0; j<rows/(knots[i].size()*strides[i]);++j){ // each cycle puts a copy of the knot vector
            for(std::size_t k=0; k<knots[i].size(); ++k){ // cycle along its elements
                for(std::size_t l=0;l<strides[i];++l){ // repeat each element
                    std::size_t node_idx = j*rows/(knots[i].size()*strides[i])+k*strides[i]+l;
                    nodes_(node_idx,i)=knots[i][k];
                    if(k==0){ // is on boundary if is the first or the last point of the knot vector
                        boundary_(node_idx)|=1;
                    }
                    else if(k==knots[i].size()-1){
                        // put a special tag to identify that one of the node coordinate is the last along some dimension
                        // which means that it has no associated element
                        boundary_(node_idx)|=3; 
                    }
                }
            }
        }
    }

    // filling elements matrix row by row
    // in each row are stored 2^M-1 nodes corresponding to the vertex of the i-th hypercube
    // all the elements can be univocally identified with a node
    // except for the boundary nodes in which at least one coordinate is the last point of a knot vector
    // for each node we only check that it can be associated to an element and then we create the correspondent element
    
    // filling the element
    // consider an M-dimensional hypercube in which each vertex is identified by M binary coordinates,
    // we call the vertex with coordinates all equal to 0 origin
    // starting from the origin we can find all the vertices simply moving in each possible combination of the M directions (#2^M-1)
    // we use this idea to find all the nodes beloging to an element
    // for each element we pick as origin the node associated to the element
    // we insert the other nodes in the element moving along the hypercube
    std::size_t element_idx = 0; // is incremented every time we add a new element 
    // if the i-th node can't be associated to an element we must change it's boundary value to 1 according to the notation
    for(std::size_t i=0; i<rows; ++i){
        if(boundary_(i)&2!=0){
            boundary_(i)=1;
        }
        else{
            // filling the i-th row 
            for(std::size_t s=0; s<(1<<M)-1;++s){
                std::size_t node_idx=i;
                    for(std::size_t t=0;t<M;++t){
                        // checking if the s-th node has a 1 in the t-th direction
                        if(s&(1<<t)!=0){
                            node_idx+=strides[t]; // we move to the next vertex via the t-th direction
                        }    
                    }
                elements_(element_idx, s)=node_idx;
            }
            element_idx++;
        }
    }
    
}

}; // namespace core
}; // namespace fdapde
  
#endif   // __MESHIGA_H__
