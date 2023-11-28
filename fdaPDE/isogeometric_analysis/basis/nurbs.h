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
#include "../../fields/scalar_expressions.h"
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

template <int N, int J>
inline auto multicontract(const Tensor<double,N-J>& weights,const SVector<N,Tensor<double,1>>& part){
    if constexpr (N==J)
        return weights;//end of recursion
    else 
        return multicontract<N,J+1>(weights.contract(part[J],Eigen::array<Eigen::IndexPair<int>,1>{}),part);
};

template <int M, int R> class Nurbs : public ScalarExpr<M,Nurbs<M,R>>{
    private:
     SVector<M,DVector<double>> knots_; // vector of vector of knots
     Tensor<double,M> weights_; // tensor of weights
     SVector<M,std::size_t> index_; // knots indexes where this basis is centered

    public:
    // constructor
     Nurbs() = default;
     Nurbs(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights,const SVector<M,std::size_t>& index) : knots_(knots), 
     weights_(weights), index_(index) {};
     Nurbs(const DVector<double>& knots, const Tensor<double,M>& weights, const SVector<M,std::size_t>& index) : 
     Nurbs(SVector<M,DVector<double>>(knots),weights,index) {};

    //evaluates the NURBS at a given point 
     inline double operator()(const SVector<M>& x) const {
        
        double num=weights_(index_);
        SVector<M,Tensor<double,1>> spline_evaluation;
        //builds the M-dim SVector containing in each position the set of spline basis evaluated
        for(std::size_t i=0;i<M;i++){
            //resize i-th tensor according to i-th weights dimension
            spline_evaluation[i].resize(weights_.dimension(i));
            //numerator update
            num=num*Spline<R>(knots_[i], index_[i])(SVector<1>(x[i]));
            //spline evaluation for i-th dimension
            for(std::size_t j=0; j<weights_.dimension(i); j++){
                spline_evaluation[i](j)=Spline<R>(knots_[i], j)(SVector<1>(x[i]));
            }
        }
        // compute the sum that appears at the denominator of the formula
        Eigen::TensorFixedSize<double, Eigen::Sizes<>> den= multicontract<M,0>(weights_,spline_evaluation);
        return num/den(0);
    };
    
};

} // namespace core
} // namespace fdapde

#endif // __NURBS_H__ 
