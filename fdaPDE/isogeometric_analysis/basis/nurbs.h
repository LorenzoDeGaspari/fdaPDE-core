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

#ifndef __NURBS_H__
#define __NURBS_H__

#include "../../splines/basis/spline.h"
#include "../../fields/scalar_field.h"
#include "../../fields/vector_field.h"
#include "../../fields/matrix_field.h"
#include "../../fields/scalar_expressions.h"
#include "../../fields/vector_expressions.h"
#include "../../fields/matrix_expressions.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace fdapde{
namespace core{

// Let u_0, u_1, ..., u_n n distinct knots. Call U = [u_0, u_1, ..., u_n] the knot vector.
// Define the splines basis N = [N_0p, ..., N_np] starting from this knot vector.
// Let w_0, ..., w_n be some positive weights
// Define the NURBS basis as R
// R_ip(x) =  [w_i/[\sum_{j} (w_j*N_jp(x))]]*N_ip(x) 

// R = nurbs order;     M = embedding dimension;
using Eigen::Tensor;

/*
    This function can be used to compute the contraction product between a tensor of order N
    and N different vectors, each of length equal to the corresponding size value of the tensor.
    For instance, if N = 2 the result will be equal to
    sum(i,j) w[i,j] * v1[i] * v2[j]

    The template parameter J is used to recall at which level of recursion we are
    External calls should be made only with J = 0
*/
template <int N, int J>
inline double multicontract(const Tensor<double,N-J>& weights,const SVector<N,Tensor<double,1>>& part){
    if constexpr (N==J)
        return weights(0);//end of recursion
    else
        // Eigen::Tensor::contract computes the tensor of order (N-J)-1 where one index is lost due to summation
        return multicontract<N,J+1>(weights.contract(part[J],Eigen::array<Eigen::IndexPair<int>,1>{}),part);
};

// this function gives a fast computation of several univariate splines, useful for pointwise evaluation of a single nurbs function
// in principle this iterative approach is faster than recursive splines computation because it avoids overhead
// from recursion and object allocation, and also because it avoid redundancy in evaluation of different splines
template <int R>
Tensor<double, 1> speval(double x, std::size_t idx, std::size_t ext, const DVector<double> & knots){

    SVector<R, double> left;
    SVector<R, double> right;

    SVector<R+1, double> N;
    N[0] = 1;

    Tensor<double, 1> result(ext);
    for(std::size_t j = 0; j < ext; ++j){
        result(j) = 0;
    }

    std::size_t i = idx;
    while(knots[i+1] <= x){
        ++i;
    }

    for(std::size_t j = 0; j < R; ++j){
        left[j] = x - knots[i-j];
        right[j] = knots[i+1+j] - x;
    }

    double s;
    double t;
    for(std::size_t j = 0; j < R; ++j){
        s = 0;
        for(std::size_t r = 0; r <= j; ++r){
            t = N[r] / (right[r] + left[j-r]);
            N[r] = s + right[r] * t;
            s = left[j-r] * t;
        }
        N[j+1] = s;
    }

    std::size_t d;
    if(i - idx >= R){
        d = i - idx - R;
        for(std::size_t j = 0; j < R+1 && j+d < ext; ++j){
            result(j + d) = N[j];
        }
    }
    else{
        d = R + idx - i;
        for(std::size_t j = d; j < R+1 && j - d < ext; ++j){
            result(j - d) = N[j];
        }
    }

    return result;

}

// this auxiliary class wraps a functor which evaluates the partial derivative of a NURBS object, in order to compute the gradient
template <int M, int R> class NurbsDerivative : public VectorExpr<M,M,NurbsDerivative<M,R>> {
    private:
     SVector<M,DVector<double>> knots_; // vector of vector of knots
     Tensor<double,M> weights_; // tensor of weights
     SVector<M,std::size_t> index_; // knots indexes where this basis is centered
     std::size_t i_; // index of the spatial coordinate along which the NURBS function is differentiated
     SVector<M, std::size_t> minIdx_;
     Eigen::array<Eigen::Index, M> extents_;
     double num0_;

    public:
    // constructor
     NurbsDerivative() = default;
     NurbsDerivative(const SVector<M,DVector<double>>& knots, const Tensor<double,M>& weights,const SVector<M,std::size_t>& index,
     std::size_t i) : knots_(knots), index_(index), i_(i) {

        for (std::size_t i = 0; i < M; ++i) {
            minIdx_[i] = (index_[i] >= R)? (index_[i]-R) : 0;
            extents_[i] = (index_[i] + R < weights.dimension(i))? (index_[i]+R+1-minIdx_[i]) : (weights.dimension(i)-minIdx_[i]);
        }
        weights_ = weights.slice(minIdx_, extents_);
        num0_ = weights(index_);

     };
     NurbsDerivative(const DVector<double>& knots, const Tensor<double,M>& weights, const SVector<M,std::size_t>& index, std::size_t i) : 
     NurbsDerivative(SVector<M,DVector<double>>(knots),weights,index,i) {};

    //evaluates the NURBS first partial derivative at a given point
    inline double operator()(const SVector<M>& x) const {

        double num = num0_; // numerator of the NURBS formula
        double num_derived; // partial derivative of num w.r.t. i-th coordinate
        SVector<M,Tensor<double,1>> spline_evaluation; // pointwise evaluation of all splines along each coordinate
        double den;// denominator of the NURBS formula
        double den_derived; // partial derivative of den w.r.t. i-th coordinate

        
        //builds the M-dim SVector containing in each position the set of spline basis evaluated
        //and compute the NURBS numerator except the i-th spline
        for(std::size_t k=0; k<M; k++){
            //resize k-th tensor according to k-th weights dimension
            spline_evaluation[k] = speval<R>(x[k], minIdx_[k], extents_[k], knots_[k]);
            if(k!=i_){
                num *= spline_evaluation[k](index_[k] - minIdx_[k]);
            }
        }

        // the derivative of the numerator is obtained by substituting the i-th spline by its derivative
        num_derived=num*Spline<R>(knots_[i_], index_[i_]).template derive<1>()(SVector<1>(x[i_]));
        // the actual numerator is just multiplied by the spline value
        num*=spline_evaluation[i_](index_[i_] - minIdx_[i_]);

        if(num == 0 && num_derived == 0)
            return 0;

        // compute the sum that appears at the denominator of the formula
        den= multicontract<M,0>(weights_,spline_evaluation);
        // by replacing the i-th evaluations with their derivatives we get the derivative of the NURBS denominator
        for(std::size_t j=0; j<extents_[i_]; j++){
            spline_evaluation[i_](j)=Spline<R>(knots_[i_], j+minIdx_[i_]).template derive<1>()(SVector<1>(x[i_]));
        }
        den_derived= multicontract<M,0>(weights_,spline_evaluation);

        //  ( N )'      N'D - ND'
        //  (---)   =  ----------
        //  ( D )         D^2
        // where f' = df/dx_i
        return (den*num_derived-den_derived*num)/(den*den);
    };
};

// this auxiliary class wraps a functor which evaluates the second partial derivative of a NURBS object,
// in order to compute the hessian matrix
template <int M, int R> class NurbsSecondDerivative : public MatrixExpr<M,M,M,NurbsSecondDerivative<M,R>> {
    private:
     SVector<M,DVector<double>> knots_; // vector of vector of knots
     Tensor<double,M> weights_; // tensor of weights
     SVector<M,std::size_t> index_; // knots indexes where this basis is centered
     std::size_t i_; // index of the spatial coordinate along which the NURBS function is differentiated
     std::size_t j_; // index of the spatial coordinate along which the NURBS function is differentiated
     SVector<M, std::size_t> minIdx_;
     Eigen::array<Eigen::Index, M> extents_;
     double num0_;

    public:
    // constructor
     NurbsSecondDerivative() = default;
     NurbsSecondDerivative(const SVector<M,DVector<double>>& knots, const Tensor<double,M>& weights,const SVector<M,std::size_t>& index,
     std::size_t i, std::size_t j) : knots_(knots), weights_(weights), index_(index), i_(i), j_(j) {

        for (std::size_t i = 0; i < M; ++i) {
            minIdx_[i] = (index_[i] >= R)? (index_[i]-R) : 0;
            extents_[i] = (index_[i] + R < weights.dimension(i))? (index_[i]+R+1-minIdx_[i]) : (weights.dimension(i)-minIdx_[i]);
        }
        weights_ = weights.slice(minIdx_, extents_);
        num0_ = weights(index_);

     };
     NurbsSecondDerivative(const DVector<double>& knots, const Tensor<double,M>& weights, const SVector<M,std::size_t>& index,
     std::size_t i, std::size_t j) : NurbsSecondDerivative(SVector<M,DVector<double>>(knots),weights,index,i,j) {};

    //evaluates the NURBS second-derivative at a given point
    inline double operator()(const SVector<M>& x) const {
        
        double num = num0_; // numerator of the NURBS formula
        double num_der_i; // partial derivative of num w.r.t. i-th coordinate
        double num_der_j; // partial derivative of num w.r.t. j-th coordinate
        double num_der_ij; // mixed partial derivative of num
        SVector<M,Tensor<double,1>> spline_evaluation; // pointwise evaluation of all splines along each coordinate
        double den; // denominator of the NURBS formula
        double den_der_i; // partial derivative of den w.r.t. i-th coordinate
        double den_der_j; // partial derivative of den w.r.t. j-th coordinate
        double den_der_ij; // mixed partial derivative of den
        
        //builds the M-dim SVector containing in each position the set of spline basis evaluated
        //and compute the NURBS numerator except the i-th and j-th splines 
        for(std::size_t k=0; k<M; k++){
            //resize i-th tensor according to i-th weights dimension
            spline_evaluation[k] = speval<R>(x[k], minIdx_[k], extents_[k], knots_[k]);
            //numerator update
            //numerator derivatives update
            if(k!=i_ && k!=j_){
                num=num*Spline<R>(knots_[k], index_[k])(SVector<1>(x[k]));
            }
        }

        // we need to distinguish the case when i and j are the same
        if(i_!=j_){ 
            num_der_i =  num * Spline<R>(knots_[i_], index_[i_]).template derive<1>()(SVector<1>(x[i_]))
                             * Spline<R>(knots_[j_], index_[j_])(SVector<1>(x[j_]));
            num_der_j =  num * Spline<R>(knots_[j_], index_[j_]).template derive<1>()(SVector<1>(x[j_]))
                             * Spline<R>(knots_[i_], index_[i_])(SVector<1>(x[i_]));
            num_der_ij = num * Spline<R>(knots_[i_], index_[i_]).template derive<1>()(SVector<1>(x[i_]))
                             * Spline<R>(knots_[j_], index_[j_]).template derive<1>()(SVector<1>(x[j_]));
            num =        num * Spline<R>(knots_[i_], index_[i_])(SVector<1>(x[i_]))
                             * Spline<R>(knots_[j_], index_[j_])(SVector<1>(x[j_]));
            if (num_der_ij == 0 && num_der_i == 0 && num_der_j == 0 && num == 0){
                return 0;
            }
            // compute the sum that appears at the denominator of the formula
            den= multicontract<M,0>(weights_,spline_evaluation);
            // replace the i-th component splines with their derivative
            for(std::size_t k=0; k<extents_[i_]; k++){
                spline_evaluation[i_](k)=Spline<R>(knots_[i_], k+minIdx_[i_]).template derive<1>()(SVector<1>(x[i_]));
            }
            den_der_i = multicontract<M,0>(weights_,spline_evaluation);
            // replace the j-th component splines with their derivative
            for(std::size_t k=0; k<extents_[j_]; k++){
                spline_evaluation[j_](k)=Spline<R>(knots_[j_], k+minIdx_[j_]).template derive<1>()(SVector<1>(x[j_]));
            }
            den_der_ij = multicontract<M,0>(weights_,spline_evaluation);
            // replace back the i-th component splines
            spline_evaluation[i_] = speval<R>(x[i_], minIdx_[i_], extents_[i_], knots_[i_]);
            den_der_j = multicontract<M,0>(weights_,spline_evaluation);
        }
        else{
            // we only multiply by one spline derivative
            num_der_i = num_der_j = num * Spline<R>(knots_[i_], index_[i_]).template derive<1>()(SVector<1>(x[i_]));
            // the mixed derivative becomes a second derivative
            num_der_ij = num * Spline<R>(knots_[i_], index_[i_]).template derive<2>()(SVector<1>(x[i_]));
            num = num * Spline<R>(knots_[i_], index_[i_])(SVector<1>(x[i_]));
            if (num_der_ij == 0 && num_der_i == 0 && num == 0){
                return 0;
            }
            // compute the sum that appears at the denominator of the formula
            den= multicontract<M,0>(weights_,spline_evaluation);
            // replace the i-th component splines with their derivative
            for(std::size_t k=0; k<extents_[i_]; k++){
                spline_evaluation[i_](k)=Spline<R>(knots_[i_], k+minIdx_[i_]).template derive<1>()(SVector<1>(x[i_]));
            }
            den_der_i = den_der_j = multicontract<M,0>(weights_,spline_evaluation);
            // replace the i-th component derivatives with the second derivative
            for(std::size_t k=0; k<extents_[i_]; k++){
                spline_evaluation[i_](k)=Spline<R>(knots_[i_], k+minIdx_[i_]).template derive<2>()(SVector<1>(x[i_]));
            }
            den_der_ij = multicontract<M,0>(weights_,spline_evaluation);
        }

        //  ( N )'°     D(N'°D - N'D° - N°D' -ND'°) + 2D'D°N
        //  (---)   =   ------------------------------------
        //  ( D )                       D^3
        // where f' = df/dx_i and f° = df/dx_j
        return (den*(num_der_ij*den - num_der_i*den_der_j - num_der_j*den_der_i - num*den_der_ij) + 2*den_der_i*den_der_j*num)
               /(den*den*den);
    };
};

template <int M, int R> class Nurbs : public ScalarExpr<M,Nurbs<M,R>>{
    private:
     SVector<M,DVector<double>> knots_; // vector of vector of knots
     Tensor<double,M> weights_; // tensor of weights
     SVector<M,std::size_t> index_; // knots indexes where this basis is centered
     VectorField<M, M, NurbsDerivative<M, R>> gradient_; //gradient
     MatrixField<M,M,M,NurbsSecondDerivative<M,R>> hessian_; //hessian
     SVector<M, std::size_t> minIdx_;
     Eigen::array<Eigen::Index, M> extents_;
     double num0_;

    public:
    // constructor
     Nurbs() = default;
     Nurbs(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights,const SVector<M,std::size_t>& index) : knots_(knots), 
     index_(index) {
        std::vector<NurbsDerivative<M, R>> gradient;
        std::array<std::array<NurbsSecondDerivative<M, R>,M>,M> hessian;
	    gradient.reserve(M);
        // define i-th element of gradient field (= partial derivative wrt i-th coordinate) 
        // define i,j-th element of the hessian field (=partial second derivative wrt i-th and j-th coordinate)
        for (std::size_t i = 0; i < M; ++i) {
            minIdx_[i] = (index_[i] >= R)? (index_[i]-R) : 0;
            extents_[i] = (index_[i] + R < weights.dimension(i))? (index_[i]+R+1-minIdx_[i]) : (weights.dimension(i)-minIdx_[i]);
            gradient.emplace_back(knots,weights,index,i);
            for(std::size_t j=i;j<M;++j){
                //using symmetry of hessian matrix since NURBS function enjoys Schwarz property where they are differentiable
                hessian[i][j]=hessian[j][i]=NurbsSecondDerivative<M, R>(knots,weights,index,i,j);
            }
        }
        // wrap the gradient components into a vectorfield
        gradient_ = VectorField<M, M, NurbsDerivative<M, R>>(gradient);
        // wrap the hessian components into a matrixfield
        hessian_= MatrixField<M,M,M,NurbsSecondDerivative<M,R>>(hessian);

        weights_ = weights.slice(minIdx_, extents_);
        num0_ = weights(index_);

     };

     Nurbs(const DVector<double>& knots, const Tensor<double,M>& weights, const SVector<M,std::size_t>& index) : 
     Nurbs(SVector<M,DVector<double>>(knots),weights,index) {};

    //evaluates the NURBS at a given point 
     inline double operator()(const SVector<M>& x) const {
        
        double num=num0_;
        SVector<M,Tensor<double,1>> spline_evaluation;
        double den;

        //builds the M-dim SVector containing in each position the set of spline basis evaluated
        // and compute the NURBS numerator
        for(std::size_t i=0;i<M;i++){
            //resize i-th tensor according to i-th weights dimension
            spline_evaluation[i] = speval<R>(x[i], minIdx_[i], extents_[i], knots_[i]);
            //numerator update
            //num=num*Spline<R>(knots_[i], index_[i])(SVector<1>(x[i]));
            num *= spline_evaluation[i](index_[i] - minIdx_[i]);
            //spline evaluation for i-th dimension
        }
        // avoid division by 0
        if(num == 0)
            return 0;
        // compute the sum that appears at the denominator of the formula
        den = multicontract<M,0>(weights_, spline_evaluation);
        return num/den;
    };

    //getters
     VectorField<M, M, NurbsDerivative<M, R>> derive () const{return gradient_;};
     MatrixField<M,M,M,NurbsSecondDerivative<M,R>> deriveTwice() const{return hessian_;}
     const SVector<M,std::size_t> & index() const { return index_; };
    
};

} // namespace core
} // namespace fdapde

#endif // __NURBS_H__ 
