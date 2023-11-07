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
template <int M, int R> class Nurbs;

// A 1D NURBS of order R centered in knot u_i.
template <int R> class Nurbs<1, R> : public ScalarExpr<1, Nurbs<1, R>>{
    private:
     DVector<double> knots_ {};   // vector of knots
     DVector<double> weights_ {};   // vector of weights
     std::size_t i_;     // knot index where this basis is centered
     ScalarField<1> denom_ = ScalarField<1>::Zero(); // store functor for the denominator in the formula
     ScalarField<1> denom_x_ = ScalarField<1>::Zero(); // store functor for the denominator x derivative
     ScalarField<1> denom_xx_ = ScalarField<1>::Zero(); // store functor for the denominator second x derivative

    public:
     // constructor
     Nurbs() = default;
     Nurbs(const DVector<double>& knots, const DVector<double>& weights, std::size_t i) : knots_(knots), weights_(weights), i_(i) {
        for(std::size_t j=0;j<knots_.rows() - R - 1;j++){
            denom_ = denom_ + weights_[j]*Spline<R>(knots_, j); // sum(j) { w_j * N_j }
            denom_x_ = denom_x_ + weights_[j]*Spline<R>(knots_, j).template derive<1>(); // sum(j) { w_j * N'_j }
            denom_xx_ = denom_xx_ + weights_[j]*Spline<R>(knots_, j).template derive<2>(); // sum(j) { w_j * N''_j }
        }
     };

    // evaluates the NURBS at a given point 
     inline double operator()(SVector<1> x) const {
        return ((weights_[i_]/denom_) * Spline<R>(knots_, i_)) (x);
     }

     // returns a functor to evaluate derivative
     //               N'_i*D - N_i*D'
     // R'_ip = w_i * -------------- , where D is the denominator in the NURBS formula
     //                     D^2
     auto derive() const {
        return (weights_[i_]/(denom_ * denom_)) *                       // w_i / D^2
               (Spline<R>(knots_, i_).template derive<1>() * denom_     // N'_i*D
               - Spline<R>(knots_, i_) * denom_x_);                     // -N_i*D'
     }

     // returns a functor to evaluate second derivative
     //                N''_i*D^2 - N_i*D*D'' - 2*N'_i*D*D' + 2*N_i*D'^2
     // R''_ip = w_i * ------------------------------------------------
     //                                      D^3
     auto deriveTwice() const {
        return (weights_[i_]/(denom_*denom_*denom_)) *                              // w_i / D^3
               (Spline<R>(knots_, i_).template derive<2>() * denom_ * denom_        // N''_i*D^2
               - Spline<R>(knots_, i_) * denom_ * denom_xx_                         // -N_i*D*D''
               - 2 * Spline<R>(knots_, i_).template derive<1>() * denom_ * denom_x_ // -2*N'_i*D*D'
               + 2 * Spline<R>(knots_, i_) * denom_x_ * denom_x_);                  // 2*N_i*D'^2
     }

};

// A 2D NURBS of order R centered in knot (u_i, v_j)
template <int R> class Nurbs<2, R> : public ScalarExpr<2, Nurbs<2, R>>{
    private:
     DVector<double> knots_x_ {};   // vector of knots 1st coordinate
     DVector<double> knots_y_ {};   // vector of knots 2nd coordinate
     DMatrix<double> weights_ {};   // vector of weights
     std::size_t i_;     // 1st knot index
     std::size_t j_;     // 2nd knot index

    public:
     // constructor
     Nurbs() = default;
     Nurbs(const DVector<double>& knots_x, const DVector<double>& knots_y, const DMatrix<double>& weights, std::size_t i, std::size_t j) : knots_x_(knots_x), knots_y_(knots_y), weights_(weights), i_(i), j_(j) {};
     Nurbs(const DVector<double>& knots, const DMatrix<double>& weights, std::size_t i, std::size_t j) : Nurbs(knots, knots, weights, i, j) {};

    // evaluates the NURBS at a given point 
     inline double operator()(SVector<2> x) const {
        double den=0.;
        // compute the sum that appears at the denominator of the formula
        for(std::size_t k=0;k<knots_x_.rows() - R - 1;k++){
            for(std::size_t l=0;l<knots_y_.rows() - R - 1;l++){
                den += weights_(k,l)*Spline<R>(knots_x_, k)(x.block<1,1>(0,0))*Spline<R>(knots_y_, l)(x.block<1,1>(1,0));
            }
        }
        return (weights_(i_,j_)/den) * Spline<R>(knots_x_, i_)(x.block<1,1>(0,0)) * Spline<R>(knots_y_, j_)(x.block<1,1>(1,0));
    }

};

} // namespace core
} // namespace fdapde

#endif // __NURBS_H__ 
