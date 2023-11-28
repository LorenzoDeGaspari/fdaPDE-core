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

template <int M, int R> class NurbsBasis {
    private:
        SVector<M,DVector<double>> knots_;
        Tensor<double,M> weights_; // tensor of weights
        std::vector<Nurbs<M,R>> basis_; 
    public:
        using const_iterator = typename std::vector<Nurbs<M,R>>::const_iterator;
        static constexpr std::size_t order = R;
        typedef Nurbs<M,R> ElementType;

        // constructor
        NurbsBasis() = default;
        NurbsBasis(const DVector<double>& knots, const Tensor<double,M>& weights) : NurbsBasis(SVector<M,DVector<double>>(knots), weights) {};
        NurbsBasis(const SVector<M,DVector<double>>& knots, const Tensor<double,M>& weights) : knots_(knots),weights_(weights) {
            // reserve space
            std::size_t n;
            // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
            for(std::size_t i=0; i < M; ++i){
                // reserve space
                n=knots_[i].size();
                knots_[i].resize(n + 2 * R);
                for (std::size_t j = 0; j < n + 2 * R; ++j) {
                if (j < R) {
                    knots_[i][j] = knots_[i][0];
                } else {
                    if (j < n + R) {
                        knots_[i][j] = knots_[i][j - R];
                    } else {
                        knots_[i][j] = knots_[i][n - 1];
                    }
                }
            }
            }
            // reserve space  
            n=0;
            for(std::size_t i=0; i< M;++i){
                n=n*(knots_[i].size()-R-1); // tensor product dim = product of dims
            }
            basis_.reserve(n); 

            // fill the basis
            SVector<M,std::size_t> index;
            for(std::size_t i = 0; i < n; ++i){
                // compute NURBS basis, knots and weights are always the same, index changes every loop
                basis_.emplace_back(knots_, weights_, index);

                // insertion is done according to lexicographical ordering
                std::size_t j = M-1;
                // increment the last index
                ++index[j];
                // when a "row" is finished, carry the increment over to the previous index 
                while(j>0 && index[j] == size(j)){
                    index[j] = 0;
                    --j;
                    ++index[j];
                }
            }
            
        }

        // getters
        const Nurbs<M,R>& operator[](std::size_t i) const { return basis_[i]; }
        const Nurbs<M,R>& operator()(const SVector<M,std::size_t> & index) const { 
            std::size_t idx=0;
            for(std::size_t i = 0; i < M; ++i){
                idx = size(i) * idx + index[i];
            }
            return basis_[idx];
        }
        int size() const { return basis_.size(); }
        int size(std::size_t i) const { return knots_[i].size() - R - 1;}
        const SVector<M,DVector<double>>& knots() const {return knots_;}
        const DVector<double>& knots(std::size_t i) const { return knots_[i]; }
        const Tensor<double,M>& weights() const { return weights_; }

        // iterators
        const_iterator begin() const { return basis_.cbegin(); }
        const_iterator end() const { return basis_.cend(); }
};
/*
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

// 2D NURBS basis
template <int R> class NurbsBasis<2, R> {
    private:
        DVector<double> knots_x_ {};
        DVector<double> knots_y_ {};
        DMatrix<double> weights_ {};
        std::vector<Nurbs<2,R>> basis_ {};
    public:
        using const_iterator = typename std::vector<Nurbs<2,R>>::const_iterator;
        static constexpr std::size_t order = R;
        typedef Nurbs<2,R> ElementType;

        // constructor
        NurbsBasis() = default;
        NurbsBasis(const DVector<double>& knots, const DMatrix<double>& weights) : NurbsBasis(knots, knots, weights) {};
        NurbsBasis(const DVector<double>& knots_x, const DVector<double>& knots_y, const DMatrix<double>& weights) : knots_x_(knots_x), knots_y_(knots_y), weights_(weights) {
            // reserve space
            std::size_t n = knots_x.size();
            knots_x_.resize(n + 2 * R);
            // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
            for (std::size_t i = 0; i < n + 2 * R; ++i) {
                if (i < R) {
                    knots_x_[i] = knots_x[0];
                } else {
                    if (i < n + R) {
                        knots_x_[i] = knots_x[i - R];
                    } else {
                        knots_x_[i] = knots_x[n - 1];
                    }
                }
            }
            // reserve space
            n = knots_y.size();
            knots_y_.resize(n + 2 * R);
            // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
            for (std::size_t i = 0; i < n + 2 * R; ++i) {
                if (i < R) {
                    knots_y_[i] = knots_y[0];
                } else {
                    if (i < n + R) {
                        knots_y_[i] = knots_y[i - R];
                    } else {
                        knots_y_[i] = knots_y[n - 1];
                    }
                }
            }
            // reserve space and compute NURBS basis
            basis_.reserve( (knots_x_.rows() - R - 1) * (knots_y_.rows() - R - 1) ); // tensor product dim = product of dims
            for (std::size_t k = 0; k < knots_x_.size() - R - 1; ++k) {
                for (std::size_t l = 0; l < knots_y_.size() - R - 1; ++l) {
                    basis_.emplace_back(knots_x_, knots_y_, weights_, k, l);   // create spline centered at (k,l)-th point of (knots_x_) x (knots_y_)
                }
            }
        }

        // getters
        const Nurbs<2,R>& operator[](std::size_t i) const { return basis_[i]; }
        const Nurbs<2,R>& operator()(std::size_t i, std::size_t j) const { return basis_[i*(knots_y_.size() - R - 1) + j]; }
        int size() const { return basis_.size(); }
        int size_x() const { return knots_x_.size() - R - 1;}
        int size_y() const { return knots_y_.size() - R - 1;}
        const DVector<double>& knots_x() const { return knots_x_; }
        const DVector<double>& knots_y() const { return knots_y_; }
        const DVector<double>& weights() const { return weights_; }

        // iterators
        const_iterator begin() const { return basis_.cbegin(); }
        const_iterator end() const { return basis_.cend(); }
};
*/
}   // namespace core
}   // namespace fdapde

#endif // __NURBS_BASIS_H__
