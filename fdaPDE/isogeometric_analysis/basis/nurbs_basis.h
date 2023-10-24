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

#ifndef __NURBS_BASIS_H__
#define __NURBS_BASIS_H__

#include "nurbs.h"

namespace fdapde {
namespace core{

// a nurbs basis of order R build over a given set of knots and weights

// forward declaration of template class, to be specialized for each value of M
// R = nurbs order;     M = embedding dimension
template <int M, int R> class NurbsBasis;

// 1D NURBS basis
template <int R> class NurbsBasis<1, R> {
    private:
        DVector<double> knots_ {};
        DVector<double> weights_ {};
        std::vector<Nurbs<1,R>> basis_ {};
    public:
        using const_iterator = typename std::vector<Nurbs<1,R>>::const_iterator;
        static constexpr std::size_t order = R;
        typedef Nurbs<1,R> ElementType;

        // constructor
        NurbsBasis() = default;
        NurbsBasis(const DVector<double>& knots, const DVector<double>& weights) : knots_(knots), weights_(weights) {
            // reserve space
            std::size_t n = knots.size();
            knots_.resize(n + 2 * R);
            // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
            for (std::size_t i = 0; i < n + 2 * R; ++i) {
                if (i < R) {
                    knots_[i] = knots[0];
                } else {
                    if (i < n + R) {
                        knots_[i] = knots[i - R];
                    } else {
                        knots_[i] = knots[n - 1];
                    }
                }
            }
            // reserve space and compute NURBS basis
            basis_.reserve(knots_.rows() - R - 1);
            for (std::size_t k = 0; k < knots_.size() - R - 1; ++k) {
                basis_.emplace_back(knots_, weights_, k);   // create spline centered at k-th point of knots_
            }
        }

        // getters
        const Nurbs<1,R>& operator[](std::size_t i) const { return basis_[i]; }
        int size() const { return basis_.size(); }
        const DVector<double>& knots() const { return knots_; }
        const DVector<double>& weights() const { return weights_; }

        // iterators
        const_iterator begin() const { return basis_.cbegin(); }
        const_iterator end() const { return basis_.cend(); }
};

}   // namespace core
}   // namespace fdapde

#endif // __NURBS_BASIS_H__
