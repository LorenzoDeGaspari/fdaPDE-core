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

#ifndef __KRONECKER_PRODUCT_H__
#define __KRONECKER_PRODUCT_H__

#include <Eigen/Core>
#include <type_traits>

#include "../utils/symbols.h"
using fdapde::internal::SparseStorage;
using fdapde::internal::DenseStorage;

namespace fdapde {
namespace core {

// Eigen-compatible implementation of the Kronecker tensor product between matrices. Use Eigen naming conventions
template <typename Derived> struct KroneckerTensorProductBase {
    // typedefs expected by eigen internals
    typedef typename Eigen::internal::traits<Derived>::Lhs Lhs;
    typedef typename Eigen::internal::traits<Derived>::Rhs Rhs;
    typedef typename Eigen::internal::ref_selector<Lhs>::type LhsNested;
    typedef typename Eigen::internal::ref_selector<Rhs>::type RhsNested;
    // required to let KroneckerProduct integration with Eigen expression templates
    typedef typename Eigen::internal::ref_selector<Derived>::type Nested;

    // expression operands
    LhsNested lhs_;
    RhsNested rhs_;
    // constructor
    KroneckerTensorProductBase(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {};
    inline Eigen::Index rows() const { return lhs_.rows() * rhs_.rows(); }
    inline Eigen::Index cols() const { return lhs_.cols() * rhs_.cols(); }
};

// dense-dense Kronecker tensor product
template <typename Lhs_, typename Rhs_>
struct KroneckerTensorProduct :
    public KroneckerTensorProductBase<KroneckerTensorProduct<Lhs_, Rhs_>>,
    public Eigen::MatrixBase<KroneckerTensorProduct<Lhs_, Rhs_>> {
    // avoid ambigous calls, refers to expected Kronecker product sizes
    using KroneckerTensorProductBase<KroneckerTensorProduct<Lhs_, Rhs_>>::rows;
    using KroneckerTensorProductBase<KroneckerTensorProduct<Lhs_, Rhs_>>::cols;
    // constructor
    KroneckerTensorProduct(const Lhs_& lhs, const Rhs_& rhs) :
        KroneckerTensorProductBase<KroneckerTensorProduct<Lhs_, Rhs_>>(lhs, rhs) {};
};

// sparse-sparse Kronecker tensor product.
template <typename Lhs_ = SpMatrix<double>,
	  typename Rhs_ = SpMatrix<double>,
	  typename LhsStorage = internal::SparseStorage,
	  typename RhsStorage = internal::SparseStorage>
struct SparseKroneckerTensorProduct :
    public KroneckerTensorProductBase<SparseKroneckerTensorProduct<Lhs_, Rhs_, LhsStorage, RhsStorage>>,
    public Eigen::SparseMatrixBase<SparseKroneckerTensorProduct<Lhs_, Rhs_, LhsStorage, RhsStorage>> {
    // avoid ambigous calls, refers to expected Kronecker product sizes
    using KroneckerTensorProductBase<SparseKroneckerTensorProduct<Lhs_, Rhs_, LhsStorage, RhsStorage>>::rows;
    using KroneckerTensorProductBase<SparseKroneckerTensorProduct<Lhs_, Rhs_, LhsStorage, RhsStorage>>::cols;
    // constructor
    SparseKroneckerTensorProduct(const Lhs_& lhs, const Rhs_& rhs) :
        KroneckerTensorProductBase<SparseKroneckerTensorProduct<Lhs_, Rhs_, LhsStorage, RhsStorage>>(lhs, rhs) {};
};

// returns the kronecker product between lhs and rhs as an eigen expression (dense-dense version)
template <typename Lhs, typename Rhs>
KroneckerTensorProduct<Lhs, Rhs> Kronecker(const Eigen::MatrixBase<Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    return KroneckerTensorProduct<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

// returns the kronecker product between lhs and rhs as an eigen expression (sparse-sparse version)
template <typename Lhs, typename Rhs>
SparseKroneckerTensorProduct<Lhs, Rhs, internal::SparseStorage, internal::SparseStorage>
Kronecker(const Eigen::SparseMatrixBase<Lhs>& lhs, const Eigen::SparseMatrixBase<Rhs>& rhs) {
    return SparseKroneckerTensorProduct<Lhs, Rhs, internal::SparseStorage, internal::SparseStorage>(
      lhs.derived(), rhs.derived());
}

}   // namespace core
}   // namespace fdapde

// definition of proper symbols in Eigen::internal namespace
namespace Eigen {
namespace internal {

// import symbols from fdapde namespace
using fdapde::core::KroneckerTensorProduct;
using fdapde::core::SparseKroneckerTensorProduct;
using fdapde::internal::SparseStorage;

// template specialization for KroneckerProduct traits (required by Eigen).
template <typename Lhs_, typename Rhs_> struct traits<KroneckerTensorProduct<Lhs_, Rhs_>> {
    typedef Lhs_ Lhs;
    typedef Rhs_ Rhs;   // export operands type
    // typedef required by eigen
    typedef typename std::decay<Lhs>::type LhsCleaned;
    typedef typename std::decay<Rhs>::type RhsCleaned;
    typedef traits<LhsCleaned> LhsTraits;
    typedef traits<RhsCleaned> RhsTraits;

    typedef Eigen::MatrixXpr XprKind;   // expression type (matrix-expression)
    // type of coefficients handled by this operator
    typedef typename ScalarBinaryOpTraits<
      typename traits<LhsCleaned>::Scalar, typename traits<RhsCleaned>::Scalar>::ReturnType Scalar;
    // storage informations
    typedef typename product_promote_storage_type<
      typename LhsTraits::StorageKind, typename RhsTraits::StorageKind, internal::product_type<Lhs, Rhs>::ret>::ret
      StorageKind;
    typedef typename promote_index_type<typename LhsTraits::StorageIndex, typename RhsTraits::StorageIndex>::type
      StorageIndex;

    enum {   // definition of required compile time informations
        Flags                = Eigen::ColMajor,
        RowsAtCompileTime    = (Lhs::RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic) ?
                                 Dynamic :
                                 Lhs::RowsAtCompileTime * Rhs::RowsAtCompileTime,
        ColsAtCompileTime    = (Lhs::ColsAtCompileTime == Dynamic || Rhs::ColsAtCompileTime == Dynamic) ?
                                 Dynamic :
                                 Lhs::ColsAtCompileTime * Rhs::ColsAtCompileTime,
        MaxRowsAtCompileTime = (Lhs::MaxRowsAtCompileTime == Dynamic || Rhs::MaxRowsAtCompileTime == Dynamic) ?
                                 Dynamic :
                                 Lhs::MaxRowsAtCompileTime * Rhs::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = (Lhs::MaxColsAtCompileTime == Dynamic || Rhs::MaxColsAtCompileTime == Dynamic) ?
                                 Dynamic :
                                 Lhs::MaxColsAtCompileTime * Rhs::MaxColsAtCompileTime
    };
};

// trait specialization for the sparse-sparse version
template <typename Lhs, typename Rhs, typename LhsStorage, typename RhsStorage>
struct traits<SparseKroneckerTensorProduct<Lhs, Rhs, LhsStorage, RhsStorage>> :
    public traits<KroneckerTensorProduct<Lhs, Rhs>> { };

// dense-dense evaluator
template <typename Lhs_, typename Rhs_>
class evaluator<KroneckerTensorProduct<Lhs_, Rhs_>> : public evaluator_base<KroneckerTensorProduct<Lhs_, Rhs_>> {
   private:
    const KroneckerTensorProduct<Lhs_, Rhs_>& xpr_;
   public:
    // typedefs expected by eigen internals
    typedef KroneckerTensorProduct<Lhs_, Rhs_> XprType;
    typedef typename nested_eval<Lhs_, 1>::type LhsNested;
    typedef typename nested_eval<Rhs_, 1>::type RhsNested;
    typedef typename XprType::CoeffReturnType CoeffReturnType;

    enum {   // required compile time constants
        CoeffReadCost = evaluator<typename std::decay<LhsNested>::type>::CoeffReadCost +
                        evaluator<typename std::decay<RhsNested>::type>::CoeffReadCost,
        Flags = Eigen::ColMajor   // only ColMajor storage orders accepted
    };
    // Kronecker product operands
    evaluator<typename std::decay<LhsNested>::type> lhs_;
    evaluator<typename std::decay<RhsNested>::type> rhs_;

    // constructor
    evaluator(const KroneckerTensorProduct<Lhs_, Rhs_>& xpr) : xpr_(xpr), lhs_(xpr.lhs_), rhs_(xpr.rhs_) {};
    // evaluate the (i,j)-th element of the kronecker product between lhs and rhs
    CoeffReturnType coeff(Eigen::Index row, Eigen::Index col) const {
        return lhs_.coeff(row / xpr_.rhs_.rows(), col / xpr_.rhs_.cols()) *
               rhs_.coeff(row % xpr_.rhs_.rows(), col % xpr_.rhs_.cols());
    }
};

// sparse-sparse evaluator
template <typename Lhs_, typename Rhs_>
class evaluator<SparseKroneckerTensorProduct<Lhs_, Rhs_, SparseStorage, SparseStorage>> :
    public evaluator_base<SparseKroneckerTensorProduct<Lhs_, Rhs_, SparseStorage, SparseStorage>> {
   private:
    const SparseKroneckerTensorProduct<Lhs_, Rhs_, SparseStorage, SparseStorage>& xpr_;
   public:
    // typedefs expected by eigen internals
    typedef SparseKroneckerTensorProduct<Lhs_, Rhs_, SparseStorage, SparseStorage> XprType;
    typedef typename evaluator<Lhs_>::InnerIterator LhsIterator;
    typedef typename evaluator<Rhs_>::InnerIterator RhsIterator;

    enum {   // required compile time constants
        CoeffReadCost = evaluator<Lhs_>::CoeffReadCost + evaluator<Rhs_>::CoeffReadCost,
        Flags         = Eigen::ColMajor   // only ColMajor storage orders accepted
    };
    // Kronecker product operands
    evaluator<Lhs_> lhs_;
    evaluator<Rhs_> rhs_;
    // rhs sizes
    Index rhs_outer_, rhs_inner_;
    inline Index nonZerosEstimate() const { return lhs_.nonZerosEstimate() * rhs_.nonZerosEstimate(); }

    // Definition of InnerIterator providing the kronecker tensor product of the operands.
    class InnerIterator {
       public:
        // usefull typedefs
        typedef typename traits<XprType>::Scalar Scalar;
        typedef typename traits<XprType>::StorageIndex StorageIndex;
        // costructor (outer is the index of the column over which we are iterating).
        InnerIterator(const evaluator<XprType>& eval, Index outer) :
            lhs_it(eval.lhs_, outer / eval.rhs_outer_),
            rhs_it(eval.rhs_, outer % eval.rhs_outer_),
            m_index(lhs_it.index() * eval.rhs_inner_ - 1),
            outer_(outer),
            eval_(eval) {
            // column of rhs detected empty, operator++ immediately returns end of iterator
            if (!rhs_it) rhs_empty = true;
            this->operator++();   // init iterator
        };

        InnerIterator& operator++() {
            if (rhs_empty) {
                m_index = -1;
                return *this;
            }
            if (rhs_it && lhs_it) {
                m_index = lhs_it.index() * eval_.rhs_inner_ + rhs_it.index();
                // (i,j)-th kronecker product value
                m_value = lhs_it.value() * rhs_it.value();
                ++rhs_it;
            } else if (!rhs_it && ++lhs_it) {                                   // start new block a_{ij}*B[,j]
                rhs_it  = RhsIterator(eval_.rhs_, outer_ % eval_.rhs_outer_);   // reset rhs iterator
                m_index = lhs_it.index() * eval_.rhs_inner_ + rhs_it.index();
                // (i,j)-th kronecker product value
                m_value = lhs_it.value() * rhs_it.value();
                ++rhs_it;
            } else {
                // end of the iterator
                m_index = -1;
            }
            return *this;
        };
        // access methods
        inline Scalar value() const { return m_value; }         // value pointed by the iterator
        inline Index col() const { return outer_; }             // current column (assume ColMajor order)
        inline Index row() const { return index(); }            // current row (assume ColMajor order)
        inline Index outer() const { return outer_; }           // outer index
        inline StorageIndex index() const { return m_index; }   // inner index
        operator bool() const { return m_index >= 0; }          // false when the iterator is over
       protected:
        // reference to operands' iterators
        LhsIterator lhs_it;
        RhsIterator rhs_it;
        const evaluator<XprType>& eval_;
        Scalar m_value;         // the value pointed by the iterator
        StorageIndex m_index;   // current inner index
        Index outer_;           // outer index as received from the constructor
       private:
        // this flag is set whenever the currently considered column of rhs is detected empty from the
        // very beginning. In this case operator++ immediately returns end-of-iterator.
        bool rhs_empty = false;
    };
    // constructor
    evaluator(const XprType& xpr) :
        xpr_(xpr),
        lhs_(xpr.lhs_),
        rhs_(xpr.rhs_),
        rhs_outer_(xpr.rhs_.outerSize()),
        rhs_inner_(xpr.rhs_.innerSize()) {};
};

}   // namespace internal
}   // namespace Eigen

#endif   // __KRONECKER_PRODUCT_H__
