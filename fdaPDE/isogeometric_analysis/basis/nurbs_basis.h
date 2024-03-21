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
                        knots_[i][j] = knots[i][0];
                    } else {
                        if (j < n + R) {
                            knots_[i][j] = knots[i][j - R];
                        } else {
                            knots_[i][j] = knots[i][n - 1];
                        }
                    }
                }
            }
            // reserve space  
            n=1;
            for(std::size_t i=0; i< M;++i){
                n=n*(knots_[i].size()-R-1); // tensor product dim = product of dims
            }
            basis_.reserve(n); 

            // fill the basis
            SVector<M,std::size_t> index;
            for(std::size_t i = 0; i < M; ++i) index[i] = 0;
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
        std::size_t index(const SVector<M,std::size_t> & multiIndex) const {
            std::size_t idx=0;
            for(std::size_t i = 0; i < M; ++i){
                idx = size(i) * idx + multiIndex[i];
            }
            return idx;
        }
        const Nurbs<M,R>& operator()(const SVector<M,std::size_t> & multiIndex) const { return basis_[index(multiIndex)]; }
        int size() const { return basis_.size(); }
        int size(std::size_t i) const { return knots_[i].size() - R - 1;}
        const SVector<M,DVector<double>>& knots() const {return knots_;}
        const DVector<double>& knots(std::size_t i) const { return knots_[i]; }
        const Tensor<double,M>& weights() const { return weights_; }

        // iterators
        const_iterator begin() const { return basis_.cbegin(); }
        const_iterator end() const { return basis_.cend(); }
};

}   // namespace core
}   // namespace fdapde

#endif // __NURBS_BASIS_H__
