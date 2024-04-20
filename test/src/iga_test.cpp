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
#include <fdaPDE/pde.h>
using fdapde::core::integrate_2D;
using fdapde::core::IntegratorTable;
using fdapde::core::GaussLegendre;
using fdapde::core::Nurbs;
using fdapde::core::NurbsBasis;
using fdapde::core::MeshIga;
using fdapde::core::Assembler;
using fdapde::core::IGALinearEllipticSolver;
using fdapde::core::ScalarField;
using fdapde::core::IGA;
using fdapde::core::IntegratorIga;
using fdapde::core::LaplaceBeltrami;
using fdapde::core::Advection;
using fdapde::core::Reaction;
using fdapde::core::PDE;
using fdapde::core::iga_order;

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


// test 1D nurbs basis derivative (functions compute the correct value)
TEST(isogeometric_analysis_test, nurbs_basis_derivative_1D){
    DVector<double> nodes;
    Eigen::Tensor<double,1> weights(7);
    nodes.resize(5);
    
    // open uniform knot vector
    for(size_t i = 0; i < 5; i++)nodes(i)=1.*i;
    // easily replicable, non trivial weights
    for(size_t i = 0; i < 7; i++)weights(i)=std::abs(std::sin(i+1));

    SpMatrix<double> expected;
    // expected results from nurbs derivative pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_test_3.mtx");

    NurbsBasis<1, 3> basis(nodes, weights);
    
    for(size_t i = 0; i < basis.size(); ++i){
        for(size_t j = 0; j < expected.cols(); ++j){
            // compare values with data from file
            EXPECT_TRUE(almost_equal(expected.coeff(i+1,j),basis[i].derive()[0](SVector<1>(expected.coeff(0,j)))));
        }
    }
}


// test 2D nurbs basis derivative (functions are accessibile and callable)
TEST(isogeometric_analysis_test, nurbs_basis_derivative_2D) {
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
    // expected results from nurbs derivative pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_test_4.mtx");

    NurbsBasis<2, 3> basis(nodes, weights);
    
    for(size_t i = 0; i < basis.size(); ++i){
        for(size_t j = 0; j < expected.cols(); ++j){
            // compare values with data from file
            EXPECT_TRUE(almost_equal(expected.coeff(2*i+2,j),basis[i].derive()[0](SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
            EXPECT_TRUE(almost_equal(expected.coeff(2*i+3,j),basis[i].derive()[1](SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
        }
    }
}

// test 1D nurbs basis second derivative (functions compute the correct value)
TEST(isogeometric_analysis_test, nurbs_basis_second_derivative_1D){
    DVector<double> nodes;
    Eigen::Tensor<double,1> weights(7);
    nodes.resize(5);
    
    // open uniform knot vector
    for(size_t i = 0; i < 5; i++)nodes(i)=1.*i;
    // easily replicable, non trivial weights
    for(size_t i = 0; i < 7; i++)weights(i)=std::abs(std::sin(i+1));

    SpMatrix<double> expected;
    // expected results from nurbs derivative pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_test_5.mtx");

    NurbsBasis<1, 3> basis(nodes, weights);
    
    for(size_t i = 0; i < basis.size(); ++i){
        for(size_t j = 0; j < expected.cols(); ++j){
            // compare values with data from file
            EXPECT_TRUE(almost_equal(expected.coeff(i+1,j),basis[i].deriveTwice()(0,0)(SVector<1>(expected.coeff(0,j)))));
        }
    }
}


// test 2D nurbs basis second derivative (functions are accessibile and callable)
TEST(isogeometric_analysis_test, nurbs_basis_second_derivative_2D) {
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
    // expected results from nurbs derivative pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_test_6.mtx");

    NurbsBasis<2, 3> basis(nodes, weights);
    
    for(size_t i = 0; i < basis.size(); ++i){
        for(size_t j = 0; j < expected.cols(); ++j){
            // compare values with data from file
            EXPECT_TRUE(almost_equal(expected.coeff(4*i+2,j),basis[i].deriveTwice()(0,0)(SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
            EXPECT_TRUE(almost_equal(expected.coeff(4*i+3,j),basis[i].deriveTwice()(1,0)(SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
            EXPECT_TRUE(almost_equal(expected.coeff(4*i+4,j),basis[i].deriveTwice()(0,1)(SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
            EXPECT_TRUE(almost_equal(expected.coeff(4*i+5,j),basis[i].deriveTwice()(1,1)(SVector<2>(expected.coeff(0,j), expected.coeff(1,j)))));
        }
    }
}

TEST(isogeometric_analysis_test, mesh_parametrization){

    SVector<3,DVector<double>> nodes;
    Eigen::Tensor<double,3> weights(2,3,2);
    Eigen::Tensor<double,4> controlpoints(2,3,2,3);
    nodes[0].resize(2);
    nodes[1].resize(3);
    nodes[2].resize(2);

    for(size_t i = 0; i < 2; i++)nodes[0](i)=1.*i;
    for(size_t i = 0; i < 3; i++)nodes[1](i)=0.5*i;
    for(size_t i = 0; i < 2; i++)nodes[2](i)=1.*i;

    for(size_t i = 0; i < 2; i++)
        for(size_t j = 0; j < 3; j++)
            for(size_t k = 0; k < 2; k++)
                weights(i,j,k) = 1.;
    
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 2; j++){
            controlpoints(0,i,j,0) = (i<2)?-1.:1.;
            controlpoints(0,i,j,1) = (i<1)?-1.:1.;
            controlpoints(0,i,j,2) = (j<1)? 0.:1.;
            controlpoints(1,i,j,0) = (i<2)? 0.:1.;
            controlpoints(1,i,j,1) = (i<1)?-1.:0.;
            controlpoints(1,i,j,2) = (j<1)? 0.:1.;
        }
    }

    SpMatrix<double> expected;
    // expected results from nurbs derivative pointwise evaluations
    Eigen::loadMarket(expected, "../data/mtx/nurbs_mesh_test.mtx");

    
    MeshIga<3,3,1> msh(nodes, weights, controlpoints);

    for(size_t j = 0; j < expected.cols(); ++j){
        // first three rows of expected contain the x-y-z coordinates of the point at which to evaluate
        SVector<3> x(expected.coeff(0,j),expected.coeff(1,j),expected.coeff(2,j));
        for(size_t i = 0; i < 3; ++i){
            EXPECT_TRUE(almost_equal(expected.coeff(3+i,j),msh.parametrization()[i](x)));
            for(size_t k = 0; k < 3; ++k){
                EXPECT_TRUE(almost_equal(expected.coeff(6+3*i+k,j),msh.gradient()(k,i)(x)));
            }
        }
    }

}

TEST(isogeometric_analysis_test, mesh_structure){

    SVector<3,DVector<double>> nodes;
    Eigen::Tensor<double,3> weights(6,6,6);
    Eigen::Tensor<double,4> controlpoints(6,6,6,3);
    nodes[0].resize(5);
    nodes[1].resize(5);
    nodes[2].resize(5);

    for(size_t i = 0; i < 5; i++)nodes[0](i)=nodes[1](i)=nodes[2](i)=1.*i;

    for(size_t i = 0; i < 6; i++)
        for(size_t j = 0; j < 6; j++)
            for(size_t k = 0; k < 6; k++)
                weights(i,j,k) = 1.;

    for(size_t i = 0; i < 6; i++){
        for(size_t j = 0; j < 6; j++){
            for(size_t k = 0; k < 6; k++){
                controlpoints(i,j,k,0) = 1.*i;
                controlpoints(i,j,k,1) = 1.*j;
                controlpoints(i,j,k,2) = 1.*k;
            }
        }
    }
    
    MeshIga<3,3,2> msh(nodes, weights, controlpoints);

    SpMatrix<double> act_nodes;
    act_nodes = msh.nodes().sparseView();
    SpMatrix<double> expected_nodes;
    Eigen::loadMarket(expected_nodes, "../data/mtx/mesh_structure/nodes.mtx");
    EXPECT_TRUE(act_nodes.isApprox(expected_nodes));

    SpMatrix<size_t> act_elements;
    act_elements = msh.elements().sparseView();
    SpMatrix<size_t> expected_elements;
    Eigen::loadMarket(expected_elements, "../data/mtx/mesh_structure/elements.mtx");
    EXPECT_TRUE(act_elements.isApprox(expected_elements));

    SpMatrix<size_t> act_neighbors;
    act_neighbors = msh.neighbors().sparseView();
    SpMatrix<size_t> expected_neighbors;
    Eigen::loadMarket(expected_neighbors, "../data/mtx/mesh_structure/neighbors.mtx");
    EXPECT_TRUE(act_neighbors.isApprox(expected_neighbors));

    SpMatrix<size_t> act_boundary;
    act_boundary = msh.boundary().sparseView();
    SpMatrix<size_t> expected_boundary;
    Eigen::loadMarket(expected_boundary, "../data/mtx/mesh_structure/boundary.mtx");
    EXPECT_TRUE(act_boundary.isApprox(expected_boundary));
}

TEST(isogeometric_analysis_test, integrator){

    SVector<3,DVector<double>> nodes;
    Eigen::Tensor<double,3> weights(4,4,4);
    Eigen::Tensor<double,4> controlpoints(4,4,4,3);
    nodes[0].resize(2);
    nodes[1].resize(2);
    nodes[2].resize(2);

    for(size_t i = 0; i < 2; i++)nodes[0](i)=nodes[1](i)=nodes[2](i)=1.*i;

    for(size_t i = 0; i < 4; i++)
        for(size_t j = 0; j < 4; j++)
            for(size_t k = 0; k < 4; k++)
                weights(i,j,k) = 1.;
    
    MeshIga<3,3,3> msh(nodes, weights, controlpoints);
    IntegratorIga<3, 3, 27> itg;
    // exact integral of x^3 + 2x^2y - 2xy^2 + 4xyz + z^3 is equal to 1
    auto f = [] (const SVector<3> & x) -> double 
        {return x[0]*x[0]*x[0] + 2*x[0]*x[0]*x[1] - 2*x[0]*x[1]*x[1] + 4*x[0]*x[1]*x[2] + x[2]*x[2]*x[2];};

    EXPECT_TRUE(almost_equal(1., itg.integrate(msh.element(0), f)));
    EXPECT_TRUE(almost_equal(1., itg.integrate(msh, f)));
}
