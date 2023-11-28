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

#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/utils.h>
#include <fdaPDE/isogeometric_analysis.h>
#include <unsupported/Eigen/SparseExtra>
using fdapde::core::integrate_2D;
using fdapde::core::IntegratorTable;
using fdapde::core::GaussLegendre;
using fdapde::core::Nurbs;
using fdapde::core::NurbsBasis;
using fdapde::core::NurbsSurface;

// tests if the integration of the constant field 1 over a quad equals its measure
TEST(isogeometric_analysis_test, integrate_constant) {
    IntegratorTable<2,4,GaussLegendre> table; // define table
    // the integral of the constant field 1 over the rectangle equals its measure
    std::function<double(SVector<2>)> f = [](SVector<2> x) -> double { return 1; };
    // test integration on the rectangle [0,1]x[0,2] ---> area = 2
    EXPECT_TRUE(almost_equal(2.0, integrate_2D(0, 1, 0, 2, f, table)));
}

// test if linear fields can be integrated over quads
TEST(isogeometric_analysis_test, integrate_linear) {
    IntegratorTable<2,4,GaussLegendre> table; // define table
    // integrate the function f(x,y) = x+y
    std::function<double(SVector<2>)> f = [](SVector<2> x) -> double { return x[0]+x[1]; };
    // test integration on the rectangle [0,1]x[0,2] ---> the solution is 3
    EXPECT_TRUE(almost_equal(3.0, integrate_2D(0, 1, 0, 2, f, table)));
}

// test if cubic fields can be integrated over quads
TEST(isogeometric_analysis_test, integrate_cubic) {
    IntegratorTable<2,4,GaussLegendre> table; // define table
    // integrate the function x^3+y^3+x^2y
    std::function<double(SVector<2>)> f = [](SVector<2> x) -> double { return x[0]*x[0]*x[0]+x[1]*x[1]*x[1]+x[0]*x[0]*x[1]; };
    // test integration on the rectangle [0,1]x[0,2] ---> the solution is 31/6
    EXPECT_TRUE(almost_equal(31.0/6.0, integrate_2D(0, 1, 0, 2, f, table)));
}

// test 1D nurbs basis (functions compute the correct value)
TEST(isogeometric_analysis_test, nurbs_basis_1D) {
    DVector<double> nodes;
    Eigen::Tensor<double,1> weights(7);
    nodes.resize(5);
    
    // open uniform knot vector
    for(size_t i = 0; i < 5; i++)nodes(i)=1.*i;
    // easily replicable, non trivial weights
    for(size_t i = 0; i < 7; i++)weights(i)=std::abs(std::sin(i+1));

    SpMatrix<double> expected;
    // expected results from nurbs pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_test_1.mtx");

    NurbsBasis<1, 3> basis(nodes, weights);

    for(size_t i = 0; i < basis.size(); ++i){
        for(size_t j = 0; j < expected.cols(); ++j){
            // compare values with data from file
            EXPECT_TRUE(almost_equal(expected.coeff(i+1,j),basis[i](SVector<1>(expected.coeff(0,j)))));
        }
    }

}

// test 2D nurbs basis (functions are accessibile and callable)
TEST(isogeometric_analysis_test, nurbs_basis_2D) {
    SVector<2,DVector<double>> nodes;
    Eigen::Tensor<double,2> weights(4,5);
    nodes[0].resize(2);
    nodes[1].resize(3);

    // open uniform knot vector
    for(size_t i = 0; i < 2; i++)nodes[0](i)=1.*i;
    for(size_t i = 0; i < 3; i++)nodes[1](i)=1.*i;
    // easily replicable, non trivial weights
    for(size_t i = 0; i < 4; i++)for(size_t j = 0; j < 5; j++)weights(i,j)=std::abs(std::sin(i+1))*std::abs(std::cos(j+1));
    

    SpMatrix<double> expected;
    // expected results from nurbs pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_test_2.mtx");

    NurbsBasis<2, 3> basis(nodes, weights);
    
    for(size_t i = 0; i < basis.size(); ++i){
        for(size_t j = 0; j < expected.cols(); ++j){
            // compare values with data from file
            EXPECT_TRUE(almost_equal(expected.coeff(i+2,j),basis[i](SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
        }
    }
}

/*
// test manifold mesh constructor
TEST(isogeometric_analysis_test, nurbs_mesh) {
    DVector<double> nodes;
    DMatrix<double> weights;
    DMatrix<DVector<double>> control_points;
    nodes.resize(3);
    weights.resize(5,5);
    control_points.resize(3,3);
    // control points for a step shape domain in 3D
    for(size_t i = 0; i < 3; i++)for(size_t j = 0; j < 3; j++){
        control_points(i,j).resize(3);
        control_points(i,j)=SVector<3>(i>=1?1:0,j-1,i>=2 ?1:0);
    }
    // uniform weight vector
    for(size_t i = 0; i < 5; i++)for(size_t j = 0; j < 5; j++)weights(i,j)=1.;
    // open uniform knot vector
    for(size_t i = 0; i < 3; i++)nodes(i)=1.*i;

    NurbsSurface<2, 3, 2> step(nodes, weights, control_points);
    for(size_t i = 0; i < step.n_nurbs(); i++){
        // check that each element can be called correctly
        step.nurbs(i)(SVector<2>(0,0));
    }
    //step.parametrization()(SVector<2>(0,0));
}
*/
