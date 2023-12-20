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

#ifndef __PARAMETRIZATION_H__
#define __PARAMETRIZATION_H__

#include "../basis/nurbs_basis.h"

namespace fdapde{
namespace core{

// in this file are the implementations of NURBS domain parametrizations, used to transform the reference domain into the curved surface domain

template <int M, int N, int R>
class ParametrizationDerivative : MatrixExpr<M,N,M,ParametrizationDerivative<M,N,R>>{

    private:
        NurbsBasis<M,R> basis_;
        Tensor<double,M> control_points_; // tensor of the control points coordinate (for each weight there is a N-dimensional control point)
        std::size_t j_; // direction along which to take the derivative

    public:

        ParametrizationDerivative() = default;
        // we only save the i-th component of the control points, so it is sufficient to chip the original ones
        ParametrizationDerivative(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M+1>& control_points, std::size_t i, std::size_t j)
         : ParametrizationDerivative(knots, weights, control_points.chip(M,i), j){ };
        ParametrizationDerivative(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M>& control_points, std::size_t j)
         : basis_(knots, weights), control_points_(control_points), j_(j) { };

        inline double operator()(const SVector<M> & x) const {
            
            double res = 0.0;
            for(auto nurb : basis_){
                res += nurb(x).derive()[j_] * control_points_(nurb.index());
            }
            return res;

        }

};

template <int M, int N, int R>
class MeshParametrization : VectorExpr<M,N,MeshParametrization<M,N,R>>{

    private:
        NurbsBasis<M,R> basis_;
        Tensor<double,M> control_points_; // tensor of the control points coordinate (for each weight there is a N-dimensional control point)

    public:

        MeshParametrization() = default;
        // we only save the i-th component of the control points, so it is sufficient to chip the original ones
        MeshParametrization(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M+1>& control_points, std::size_t i)
        : MeshParametrization(knots, weights, control_points.chip(M,i)) { };
        MeshParametrization(const SVector<M,DVector<double>>& knots,const Tensor<double,M>& weights, const Tensor<double,M>& control_points)
        : basis_(knots, weights), control_points_(control_points) { };

        inline double operator()(const SVector<M> & x) const {
            
            double res = 0.0;
            for(auto nurb : basis_){
                res += nurb(x) * control_points_(nurb.index());
            }
            return res;

        }

};

}; // namespace core
}; // namespace fdapde

#endif // __PARAMETRIZATION_H__