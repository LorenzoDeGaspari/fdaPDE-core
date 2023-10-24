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

namespace fdapde{
namespace core{

// Let u_0, u_1, ..., u_n n distinct knots. Call U = [u_0, u_1, ..., u_n] the knot vector.
// Define the splines basis N = [N_0p, ..., N_np] starting from this knot vector.
// Let w_0, ..., w_n be some positive weights
// Define the NURBS basis as R
// R_ip(x) =  [w_i/[\sum_{j} (w_j*N_jp(x))]]*N_ip(x)

// forward declaration of template class, to be specialized for each value of M
// R = nurbs order;     M = embedding dimension
template <int R, int M> class Nurbs;

// A 1D NURBS of order R centered in knot u_i.
template <int R> class Nurbs<1, R> : public ScalarExpr<1, Nurbs<1, R>>{
    private:
     DVector<double> knots_ {};   // vector of knots
     DVector<double> weights_ {};   // vector of weights
     std::size_t i_;     // knot index where this basis is centered

    public:
     // constructor
     Nurbs() = default;
     Nurbs(const DVector<double>& knots, const DVector<double>& weights, std::size_t i) : knots_(knots), weights_(weights), i_(i) {};

    // evaluates the NURBS at a given point 
     inline double operator()(SVector<1> x) const {
        double den=0.;
        // compute the sum that appears at the denominator of the formula
        for(std::size_t j=0;j<knots_.rows() - R - 1;j++){
            den += weights_[j]*Spline<R>(knots_, j)(x);
        }
        return (weights_[i_]/den) * Spline<R>(knots_, i_)(x);
    }

};


} // namespace core
} // namespace fdapde

#endif // __NURBS_H__ 
