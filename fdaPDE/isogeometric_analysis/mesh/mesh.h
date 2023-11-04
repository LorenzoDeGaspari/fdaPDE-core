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

#ifndef __NURBSDOMAIN_H__
#define __NURBSDOMAIN_H__

#include "../basis/nurbs_basis.h"
#include "../../fields.h"

namespace fdapde{
namespace core{

// M local space dimension, N embedding space dimension, R NURBS order
template <int M, int N,int R> class NurbsDomain;

// partial specialization for M=2, N=3
template <int R>
class NurbsDomain<2,3,R> : public VectorExpr<2,3,NurbsDomain<2,3,R>> {

   protected:
    // Nurbs basis
    NurbsBasis<2,R> basis_;
    // control points
    DMatrix<DVector<double>> control_points_;

   public:
    NurbsDomain() = default;
    NurbsDomain(const DVector<double>& knots, const DMatrix<double>& weights, 
    const DMatrix<DVector<double>>& control_points) : NurbsDomain(knots, knots, weights, control_points) { };
    NurbsDomain(const DVector<double>& knots_x, const DVector<double>& knots_y,
    const DMatrix<double>& weights, const DMatrix<DVector<double>>& control_points) : basis_(knots_x,knots_y,weights), 
    control_points_(control_points) { };

    // getters
    const Nurbs<2, R>& nurbs(int ID) const { return basis_[ID]; };
    int n_basis() const { return basis_.size(); };
    const DMatrix<DVector<double>>& control_points() const{ return control_points_; };
    DMatrix<DVector<double>>& control_points() { return control_points_; };

    // call operator for the domain parametrization
    SVector<3> operator()(const SVector<2> & x) const;

};

// evaluate starting from the point x in the parametric domain
// its corresponding point on the manifold
template <int R>
SVector<3> NurbsDomain<2,3,R>::operator()(const SVector<2> & x) const{

    SVector<3> ret(0.,0.,0.);
    // for each basis function
    for(std::size_t i = 0; i < basis_.size_x(); ++i){
        for(std::size_t j = 0; j < basis_.size_y(); ++j){
            // for each component
            for(std::size_t k = 0; k < 3; ++k){
                ret[k] += basis_(i,j)(x) * control_points_(i,j)[k];
            }
        }
    }
    return ret;

}

}; // namespace core
}; // namespace fdapde
  
#endif   // __NURBSDOMAIN_H__
