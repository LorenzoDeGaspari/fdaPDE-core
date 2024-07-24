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

        using ParametrizationType = VectorField<M, N, MeshParametrization<M,N,R>>;
        using GradientType = MatrixField<M, N, M, ParametrizationDerivative<M,N,R>>;

        DVector<std::size_t> functions_; // indexes of the basis functions that have support in the element
        std::size_t ID_; // id of the element
        std::shared_ptr<ParametrizationType> parametrization_;
        std::shared_ptr<GradientType> gradient_;
        SVector<M> left_coords_;
        SVector<M> right_coords_;
        double measure_;
        double integral_measure_; // this value is equal to measure_ / 2^M, it is useful for LGL integrals

    public:

        ElementIga() = default;
        ElementIga(const DVector<std::size_t> & functions, std::size_t ID, 
        const ParametrizationType & parametrization, const GradientType & gradient, const SVector<M> & left_coords, const SVector<M> & right_coords) 
        : functions_(functions), ID_(ID), parametrization_(std::make_shared<ParametrizationType>(parametrization)),
        gradient_(std::make_shared<GradientType>(gradient)), left_coords_(left_coords), right_coords_(right_coords)  {
            measure_ = 1;
            for(std::size_t i = 0; i < M; ++i){
                measure_ *= right_coords_[i] - left_coords_[i];
            }
            integral_measure_ = measure_ / (1<<M);
        }
        const ParametrizationType& parametrization() const { return *parametrization_; }
        const GradientType& gradient() const { return *gradient_; };

        std::size_t ID() const { return ID_; }
        const DVector<std::size_t> & functions() const { return functions_; }
        std::size_t n_functions() const { return functions_.rows(); }
        // overload operator[] to get directly the i-th index
        std::size_t operator[] (std::size_t i) const { return functions_[i]; }
        const SVector<N> & left_coords() const { return left_coords_; }
        const SVector<N> & right_coords() const { return right_coords_; }
        double measure() const { return measure_; }
        double integral_measure() const { return integral_measure_; }
        SVector<M> affine_map(const SVector<M> & p) const {
            SVector<M> x;
            for(std::size_t i = 0; i < M; ++i){
                x[i] = 0.5*(right_coords_[i] + left_coords_[i] + (right_coords_[i] - left_coords_[i]) * p[i]);
            }
            return x;
        }

};

// N local space dimension, M embedding space dimension
template <int M, int N, int R> class MeshIga{

    protected:

        SVector<M,DVector<double>> knots_;
        Tensor<double,M> weights_; // tensor of weights
        Tensor<double,M+1> control_points_; // tensor of the control points (for each weight there is a N-dimensional control point)
        VectorField<M, N, MeshParametrization<M,N,R>> parametrization_;
        MatrixField<M,N,M,ParametrizationDerivative<M,N,R>> gradient_;
        NurbsBasis<M, R> basis_;

        DMatrix<double> nodes_ {};
        DMatrix<std::size_t> boundary_ {};
        DMatrix<std::size_t, Eigen::RowMajor> elements_ {};
        DMatrix<std::size_t> neighbors_ {};
        std::vector<ElementIga<M,N,R>> elements_cache_;
        DMatrix<std::size_t> boundary_dofs_ {};

    public:

        typedef VectorField<M, N, MeshParametrization<M,N,R>> ParametrizationType;
        typedef MatrixField<M,N,M,ParametrizationDerivative<M,N,R>> DerivativeType;

        MeshIga() = default;
        MeshIga(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M+1>& control_points);

        const VectorField<M, N, MeshParametrization<M,N,R>>& parametrization() const { return parametrization_; }
        const MatrixField<M,N,M,ParametrizationDerivative<M,N,R>>& gradient() const { return gradient_; }
        const NurbsBasis<M, R> & basis() const { return basis_; }

        const ElementIga<M,N,R> & element(std::size_t ID) const { return elements_cache_[ID]; }
        ElementIga<M,N,R>& element(std::size_t ID) { return elements_cache_[ID]; }
        SVector<N> node(int ID) const { return nodes_.row(ID); }
        bool is_on_boundary(int ID) const { return boundary_(ID) == 1; }
        const DMatrix<double>& nodes() const { return nodes_; }
        const DMatrix<std::size_t, Eigen::RowMajor>& elements() const { return elements_; }
        const DMatrix<std::size_t> & neighbors() const { return neighbors_; }
        const DMatrix<std::size_t>& boundary() const { return boundary_; }
        const DMatrix<std::size_t>& boundary_dofs() const { return boundary_dofs_; }
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

        enum {
            local_dimension = M,
            embedding_dimension = N,
        };

};

template <int M, int N, int R>
MeshIga<M,N,R>::MeshIga(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M+1>& control_points) :
    knots_(knots), weights_(weights), control_points_(control_points), basis_(knots, weights) {

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
    parametrization_ = VectorField<M, N, MeshParametrization<M,N,R>>(param);
    // wrap the gradient components into a matrixfield
    gradient_ = MatrixField<M,N,M,ParametrizationDerivative<M,N,R>>(grad);

    // setting the dimensions of nodes, boundary and elements matrices
    std::size_t rows = 1;
    std::size_t element_rows = 1;
    std::size_t tmp = 1;
    std::size_t tmp_el = 1;
    SVector<M,std::size_t> strides;
    SVector<M+1,std::size_t> element_strides;
    for (std::size_t i=0; i<M; ++i){
        rows *= knots[i].size();
        element_rows *= (knots[i].size()-1);
        strides[i] = tmp;
        element_strides[i] = tmp_el;
        tmp *= knots[i].size();
        tmp_el *= knots[i].size()-1;
    }
    element_strides[M] = element_rows;
    nodes_.resize(rows,M);
    boundary_.resize(rows,1);
    elements_.resize(element_rows,(1<<M));
    neighbors_ = DMatrix<std::size_t>::Constant(element_rows, 2*M, -1);

    // filling the nodes matrix with the cartesian product of knots
    // filling column by column
    // each column is formed by repeating over and over the i-th knot vector
    // where each of its elements is repeated *stride* times (changes at each cycle)
    // this ensures that all the possible tuples of knots are considered
    // boundary points are the ones in which at least one component is the first or the last point of a knot vector
    for(std::size_t i = 0; i < rows; ++i){
        boundary_(i) = 0;
    }
    for(std::size_t i=0; i<M; ++i){ // cycle over columns
        for(std::size_t j=0; j<rows/(knots[i].size()*strides[i]);++j){ // each cycle puts a copy of the knot vector
            for(std::size_t k=0; k<knots[i].size(); ++k){ // cycle along its elements
                for(std::size_t l=0;l<strides[i];++l){ // repeat each element
                    std::size_t node_idx = j*(knots[i].size()*strides[i])+k*strides[i]+l;
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
        if((boundary_(i)&2)!=0){
            boundary_(i)=1;
        }
        else{
            // filling the i-th row 
            for(std::size_t s=0; s<(1<<M);++s){
                std::size_t node_idx=i;
                    for(std::size_t t=0;t<M;++t){
                        // checking if the s-th node has a 1 in the t-th direction
                        if((s&(1<<t))!=0){
                            node_idx+=strides[t]; // we move to the next vertex via the t-th direction
                        }    
                    }
                elements_(element_idx, s)=node_idx;
            }
            element_idx++;
        }
    }

    // each element's neighbors are elements who have the same indexes along each dimension
    // except for one dimension where the index is the previous or the next one (if they exist)
    // w.r.t. the considered element
    // to simplify the computation, we add the "next" element along a direction to the current one's neighbor list
    // while simultaneously adding the current one to the "next" one's list
    for(std::size_t i = 0; i < element_rows; ++i){
        for(std::size_t j = 0; j < M; ++j){
            std::size_t k = 0;
            // we need to look for the first uninitialized element
            // note: it may not be zero if this element was the successor of one that came before
            while(neighbors_(i, k) != -1){
                ++k;
            }
            // this condition ensures that the element we are considering is not the last one of its row
            // along the j-th dimension, i.e. that the "next" one is not out of bounds
            if(!((i+element_strides[j]) % element_strides[j+1] < element_strides[j])){
                neighbors_(i,k++) = i + element_strides[j];
                std::size_t l = 0;
                // we need to look for the first uninitialized element
                while(neighbors_(i + element_strides[j], l) != -1){
                    ++l;
                }
                neighbors_(i + element_strides[j], l) = i;
            }
        }
    }

    // populate elements_cache_, the id, parametrization and gradient are easy to obtain,
    // but we must construct the list of indexes of functions that have support on the element
    // for each dimension there are R+1 basis functions that are not null between two knots,
    // so we have (R+1)^M different combinations that we find by cycling along a multi-index
    elements_cache_.resize(element_rows);
    SVector<M, std::size_t> elMultiIndex;
    for(std::size_t j = 0; j < M; ++j) { elMultiIndex[j] = 0; };

    std::size_t fnSize = pow(R+1, M);
    DVector<std::size_t> functions;
    functions.resize(fnSize);

    for(std::size_t i = 0; i < element_rows; ++i){

        SVector<M, std::size_t> fnMultiIndex(elMultiIndex);

        for(std::size_t j = 0; j < fnSize; ++j){

            functions[j] = basis_.index(fnMultiIndex);

            // increment the inner multi-index and perform "carry" operations if necessary
            std::size_t k = 0;
            ++fnMultiIndex[k];
            while(k<M-1 && fnMultiIndex[k] > elMultiIndex[k] + R){
                fnMultiIndex[k] = elMultiIndex[k];
                ++k;
                ++fnMultiIndex[k];
            }
        }

        elements_cache_[i] = ElementIga<M,N,R>(functions, i, parametrization_, gradient_, nodes_.row(elements_.row(i)(0)), 
                                               nodes_.row(elements_.row(i)((1<<M)-1)));

        // increment the element multi-index and perform "carry" operations if necessary
        std::size_t k = 0;
        ++elMultiIndex[k];
        // the comparison with the knots size is backwards because the function multi-indexes are col-major
        while(k<M-1 && elMultiIndex[k] >= knots[k].size()-1){
            elMultiIndex[k] = 0;
            ++k;
            ++elMultiIndex[k];
        }

    }

    // recover information on which functions are not identically null on the boundary of the domain
    // unlike the FEM lagrangian basis, this is not a trivial information to gather from nodes info
    // first compute how many are there
    std::size_t n_boundary_dofs = 0;
    std::size_t tmp_bd = 2;
    for(std::size_t i = 1; i < M; ++i){
        // for each dimension we can couple the first and last (2) univariate B-Splines
        // with any other from the other dimensions
        tmp_bd *= weights_.dimension(i);
    }
    n_boundary_dofs += tmp_bd;
    for(std::size_t i = 0; i < M-1; ++i){
        // compute in 2 lines to avoid integer division
        tmp_bd *= weights_.dimension(i) - 2; // the -2 is to not count twice the functions that touch the boundary in more dimensions
        tmp_bd /= weights_.dimension(i+1);
        n_boundary_dofs += tmp_bd;
    }

    boundary_dofs_.resize(n_boundary_dofs,1);

    // cycle all functions
    SVector<M, std::size_t> fnMultiIndex;
    for(std::size_t i = 0; i < M; ++i){
        fnMultiIndex[i] = 0;
    }
    for(std::size_t i = 0; i < n_boundary_dofs;){ // increment only if the function is added to the boundary list

        bool is_on_bd = false;
        // check for a specific function if it is on boundary
        for(std::size_t j = 0; j < M && !is_on_bd; ++j){
            if(fnMultiIndex[j] == 0 || fnMultiIndex[j] == weights_.dimension(j) - 1){
                is_on_bd = true;
            }
        }

        if(is_on_bd){
            boundary_dofs_(i) = basis_.index(fnMultiIndex);
            ++i; // increment only if the function is added to the boundary list
        }

        // increment the inner multi-index and perform "carry" operations if necessary
        std::size_t k = M-1;
        ++fnMultiIndex[k];
        while(k>0 && fnMultiIndex[k] >= weights_.dimension(k) ){
            fnMultiIndex[k] = 0;
            --k;
            ++fnMultiIndex[k];
        }

    }
    

}


}; // namespace core
}; // namespace fdapde
  
#endif   // __MESHIGA_H__
