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

#ifndef __NURBSSURFACE_H__
#define __NURBSSURFACE_H__

#include <Eigen/Core>
#include <array>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../../utils/combinatorics.h"
#include "../../utils/symbols.h"
#include "../basis/nurbs_basis.h"
#include "../../fields.h"

namespace fdapde{
namespace core{

// M local space dimension, N embedding space dimension, R NURBS order
template <int M, int N,int R> class NurbsSurface;

// partial specialization for M=2, N=3
template <int R>
class NurbsSurface<2,3,R> {
   protected:
    // Nurbs  
    NurbsBasis<2,R> nurbs_;
    // control points
    DMatrix<DVector<double>> control_points_;
    // callable object representing the NURBS-parametrization
    using Parametrization = VectorField<2,3>;
    Parametrization surface_;

    public:
    NurbsSurface() = default;
    NurbsSurface(const DVector<double>& knots, const DMatrix<double>& weights, 
    const DMatrix<DVector<double>>& control_points) : NurbsSurface(knots, knots, weights, control_points) { };
    NurbsSurface(const DVector<double>& knots_x, const DVector<double>& knots_y, const DMatrix<double>& weights, 
    const DMatrix<DVector<double>>& control_points);

    // getters
    const Nurbs<2, R>& nurbs(int ID) const { return nurbs_[ID]; };    
    int n_nurbs() const { return nurbs_.size(); };
    const DMatrix<DVector<double>>& control_points() const{ return control_points_; };
    DMatrix<DVector<double>>& control_points() { return control_points_; };
    const Parametrization & parametrization() const{ return surface_; };
};


//implementative details
//constructor
template <int R>
NurbsSurface<2,3,R>::NurbsSurface(const DVector<double>& knots_x, const DVector<double>& knots_y, 
    const DMatrix<double>& weights, const DMatrix<DVector<double>>& control_points) : nurbs_(knots_x,knots_y,weights), 
    control_points_(control_points){

        // TODO generate expression for parametrization

};

}; // namespace core
}; // namespace fdapde
  
#endif   // __NURBSSURFACE_H__
