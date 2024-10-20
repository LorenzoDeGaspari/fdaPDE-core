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

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <fdaPDE/utils.h>
#include <fdaPDE/isogeometric_analysis.h>
#include <unsupported/Eigen/SparseExtra>
#include <fdaPDE/pde.h>

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

struct testData{

    public:
        std::string name_;
        std::size_t ref_n_;

        std::string kx_filename_;
        std::string ky_filename_;
        std::string weights_filename_;
        std::string cpx_filename_;
        std::string cpy_filename_;
        
        DMatrix<double> solution_;
        std::size_t assembler_time_;
        std::size_t solver_time_;

        testData(std::string name, std::size_t ref_n, std::string kx_filename, std::string ky_filename,
            std::string weights_filename, std::string cpx_filename, std::string cpy_filename)
            : name_(name), ref_n_(ref_n), kx_filename_(kx_filename), ky_filename_(ky_filename), weights_filename_(weights_filename),
              cpx_filename_(cpx_filename), cpy_filename_(cpy_filename) {};

};

void solve_problem(testData & test);
std::size_t dbg_cnt = 0;
void deb(){
    std::cout << dbg_cnt++ << "\n";
}

int main(){
    
    testData test_ref0("Refinement 0", 36, "data/quad_kx_ref0.mtx", "data/quad_ky_ref0.mtx",
                       "data/quad_weights_ref0.mtx", "data/quad_cpx_ref0.mtx", "data/quad_cpy_ref0.mtx");

    testData test_ref1("Refinement 1", 64, "data/quad_kx_ref1.mtx", "data/quad_ky_ref1.mtx",
                       "data/quad_weights_ref1.mtx", "data/quad_cpx_ref1.mtx", "data/quad_cpy_ref1.mtx");

    testData test_ref2("Refinement 2", 144, "data/quad_kx_ref2.mtx", "data/quad_ky_ref2.mtx",
                       "data/quad_weights_ref2.mtx", "data/quad_cpx_ref2.mtx", "data/quad_cpy_ref2.mtx");

    testData test_ref3("Refinement 3", 324, "data/quad_kx_ref3.mtx", "data/quad_ky_ref3.mtx",
                       "data/quad_weights_ref3.mtx", "data/quad_cpx_ref3.mtx", "data/quad_cpy_ref3.mtx");

    std::vector<testData> all_tests({test_ref0, test_ref1, test_ref2, test_ref3});

    for(auto test : all_tests){
        solve_problem(test);
    }

    return 0;
}

void solve_problem(testData & test){

    // sparse matrix to read from market files
    SpMatrix<double> tmp_sp;

    SVector<2,DVector<double>> knots;

    // load x coordinate knot vector
    Eigen::loadMarket(tmp_sp, test.kx_filename_);
    knots[0].resize(tmp_sp.cols());
    for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
        knots[0][j] = tmp_sp.coeff(0,j);
    }

    // load y coordinate knot vector
    Eigen::loadMarket(tmp_sp, test.ky_filename_);
    knots[1].resize(tmp_sp.cols());
    for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
        knots[1][j] = tmp_sp.coeff(0,j);
    }  

    // load weights
    Eigen::loadMarket(tmp_sp, test.weights_filename_);
    Eigen::Tensor<double,2> weights(tmp_sp.rows(),tmp_sp.cols());
    for(std::size_t i = 0; i < tmp_sp.rows(); ++i){
        for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
            tmp_sp.coeffRef(i,j) = 1.;
            weights(i,j) = tmp_sp.coeff(i,j);
        }
    }
    Eigen::saveMarket(tmp_sp, test.weights_filename_);
    
    // control points have the same dimensions as weights, plus a dimension for the axis (size 2: (x,y))
    Eigen::Tensor<double,3> control_points(tmp_sp.rows(),tmp_sp.cols(), 2);

    // load control points x coordinate
    Eigen::loadMarket(tmp_sp, test.cpx_filename_);
    for(std::size_t i = 0; i < tmp_sp.rows(); ++i){
        for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
            control_points(i,j,0) = tmp_sp.coeff(i,j);
        }
    }

    // load control points x coordinate
    Eigen::loadMarket(tmp_sp, test.cpy_filename_);
    for(std::size_t i = 0; i < tmp_sp.rows(); ++i){
        for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
            control_points(i,j,1) = tmp_sp.coeff(i,j);
        }
    }

    // create a mesh
    fdapde::core::MeshIga<2,2,2> mesh(knots, weights, control_points);

    // create a diffusion operator
    Eigen::Matrix2d coeff;
    coeff(0,0) = coeff(1,1) = 1.;
    coeff(1,0) = coeff(0,1) = 0.;
    fdapde::core::Diffusion <fdapde::core::IGA, decltype(coeff)> K(coeff);
    
    // wrap forcing term in a scalar field
    auto f = [] (const SVector<2> & x) -> double { return sin(2*M_PI*x[0]) * (4*M_PI*M_PI*x[1]*x[1] - 4*M_PI*M_PI*x[1]-2);};
    fdapde::core::ScalarField<2> ff;
    ff = f;

    // create a pde object
    fdapde::core::PDE<decltype(mesh),decltype(K), decltype(ff), fdapde::core::IGA, fdapde::core::iga_order<2>> pde_r0(mesh,K,ff);
    // dummy object for dirchlet bc (we only support homogeneous, but we pass from the PDE interface)
    DMatrix<double> d_bc;
    d_bc.resize(2,2);
    
    const auto t0 = std::chrono::high_resolution_clock::now();
    // assemble the matrices
    pde_r0.init();
    // enforce strong boundary conditions
    pde_r0.set_dirichlet_bc(d_bc);
    const auto t_ass = std::chrono::high_resolution_clock::now();
    // solve the problem
    pde_r0.solve();
    const auto t_sol = std::chrono::high_resolution_clock::now();

    test.assembler_time_ = (std::chrono::duration_cast<std::chrono::milliseconds>(t_ass-t0)).count();
    test.solver_time_ = (std::chrono::duration_cast<std::chrono::microseconds>(t_sol-t_ass)).count();

    std::cout << "Test: " << test.name_ << "\n";
    std::cout << "Refinement: " << test.ref_n_ << "\n";
    std::cout << "Assembler: " << test.assembler_time_ << "ms\n";
    std::cout << "Solver: " << test.solver_time_ << "us\n";

}