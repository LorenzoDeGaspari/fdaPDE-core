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
using fdapde::core::integrate_2D;
using fdapde::core::IntegratorTable;
using fdapde::core::GaussLegendre;

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

// test if linear fields can be integrated over quads
TEST(isogeometric_analysis_test, integrate_cubic) {
    IntegratorTable<2,4,GaussLegendre> table; // define table
    // integrate the function x^3+y^3+x^2y
    std::function<double(SVector<2>)> f = [](SVector<2> x) -> double { return x[0]*x[0]*x[0]+x[1]*x[1]*x[1]+x[0]*x[0]*x[1]; };
    // test integration on the rectangle [0,1]x[0,2] ---> the solution is 31/6
    EXPECT_TRUE(almost_equal(31.0/6.0, integrate_2D(0, 1, 0, 2, f, table)));
}
