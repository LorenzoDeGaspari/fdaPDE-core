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
        std::string cpz_filename_;
        std::string solution_geopde_filename_;
        
        DMatrix<double> solution_;
        std::size_t assembler_time_;
        std::size_t solver_time_;

        double err_L2_;
        double err_L_inf_;
        double err_H1_;

        testData(std::string name, std::size_t ref_n, std::string kx_filename, std::string ky_filename,
            std::string weights_filename, std::string cpx_filename, std::string cpy_filename, std::string cpz_filename,
            std::string geopde_sol_filename)
            : name_(name), ref_n_(ref_n), kx_filename_(kx_filename), ky_filename_(ky_filename), weights_filename_(weights_filename),
              cpx_filename_(cpx_filename), cpy_filename_(cpy_filename), cpz_filename_(cpz_filename),
              solution_geopde_filename_(geopde_sol_filename){};

};

void solve_problem(testData & test);
void post_processing(std::vector<testData> & all_test, std::string  geopde_times_filename);
std::size_t dbg_cnt = 0;
void deb(){
    std::cout << dbg_cnt++ << "\n";
}

int main(){
    
    testData test_ref0("Refinement 0", 33, "data/curlyplate_kx_ref0.mtx", "data/curlyplate_ky_ref0.mtx",
                       "data/curlyplate_weights_ref0.mtx", "data/curlyplate_cpx_ref0.mtx",
                       "data/curlyplate_cpy_ref0.mtx", "data/curlyplate_cpz_ref0.mtx", "data/curlyplate_geopde_sol_33.mtx");

    testData test_ref1("Refinement 1", 56, "data/curlyplate_kx_ref1.mtx", "data/curlyplate_ky_ref1.mtx",
                       "data/curlyplate_weights_ref1.mtx", "data/curlyplate_cpx_ref1.mtx",
                       "data/curlyplate_cpy_ref1.mtx", "data/curlyplate_cpz_ref1.mtx", "data/curlyplate_geopde_sol_56.mtx");

    testData test_ref2("Refinement 2", 108, "data/curlyplate_kx_ref2.mtx", "data/curlyplate_ky_ref2.mtx",
                       "data/curlyplate_weights_ref2.mtx", "data/curlyplate_cpx_ref2.mtx",
                       "data/curlyplate_cpy_ref2.mtx", "data/curlyplate_cpz_ref2.mtx", "data/curlyplate_geopde_sol_108.mtx");

    testData test_ref3("Refinement 3", 240, "data/curlyplate_kx_ref3.mtx", "data/curlyplate_ky_ref3.mtx",
                       "data/curlyplate_weights_ref3.mtx", "data/curlyplate_cpx_ref3.mtx",
                       "data/curlyplate_cpy_ref3.mtx", "data/curlyplate_cpz_ref3.mtx", "data/curlyplate_geopde_sol_240.mtx");

    std::vector<testData> all_tests({test_ref0, test_ref1, test_ref2 ,test_ref3});

    for(std::size_t i = 0; i < all_tests.size(); ++i){
        solve_problem(all_tests[i]);
    }

    post_processing(all_tests, "data/curlyplate_geopde_times.mtx");
    return 0;
}

void solve_problem(testData & test){

    std::cout << "Start test " << test.name_ << "\n";
    std::cout << "Loading data from files...\n";
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
    Eigen::Tensor<double,3> control_points(tmp_sp.rows(),tmp_sp.cols(), 3);

    // load control points x coordinate
    Eigen::loadMarket(tmp_sp, test.cpx_filename_);
    for(std::size_t i = 0; i < tmp_sp.rows(); ++i){
        for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
            control_points(i,j,0) = tmp_sp.coeff(i,j);
        }
    }

    // load control points y coordinate
    Eigen::loadMarket(tmp_sp, test.cpy_filename_);
    for(std::size_t i = 0; i < tmp_sp.rows(); ++i){
        for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
            control_points(i,j,1) = tmp_sp.coeff(i,j);
        }
    }

    // load control points z coordinate
    Eigen::loadMarket(tmp_sp, test.cpz_filename_);
    for(std::size_t i = 0; i < tmp_sp.rows(); ++i){
        for(std::size_t j = 0; j < tmp_sp.cols(); ++j){
            control_points(i,j,2) = tmp_sp.coeff(i,j);
        }
    }
    std::cout << "DONE!\n";
    std::cout << "Creating mesh object...\n";
    // create a mesh
    fdapde::core::MeshIga<2,3,2> mesh(knots, weights, control_points);
    std::cout << "DONE!\n";
    std::cout << "Creating operator object...\n";
    // create a diffusion operator
    Eigen::Matrix2d coeff;
    coeff(0,0) = coeff(1,1) = -1.;
    coeff(1,0) = coeff(0,1) = 0.;
    fdapde::core::Diffusion <fdapde::core::IGA, decltype(coeff)> K(coeff);
    std::cout << "DONE!\n";
    std::cout << "Creating forcing term object...\n";
    // wrap forcing term in a scalar field
    auto f = [] (const SVector<3> & x) -> double { return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];};
    fdapde::core::ScalarField<3> ff;
    ff = f;
    std::cout << "DONE!\n";
    std::cout << "Creating PDE object...\n";
    // create a pde object
    fdapde::core::PDE<decltype(mesh),decltype(K), decltype(ff), fdapde::core::IGA, fdapde::core::iga_order<2>> pde_r0(mesh,K,ff);
    std::cout << "DONE!\n";
    std::cout << "Assembling algebraic terms...";
    // dummy object for dirchlet bc (we only support homogeneous, but we pass from the PDE interface)
    DMatrix<double> d_bc;
    d_bc.resize(2,2);
    
    const auto t0 = std::chrono::high_resolution_clock::now();
    // assemble the matrices
    pde_r0.init();
    // enforce strong boundary conditions
    pde_r0.set_dirichlet_bc(d_bc);
    const auto t_ass = std::chrono::high_resolution_clock::now();
    std::cout << "DONE!\n";
    std::cout << "Solving linear system...\n";
    const auto t1 = std::chrono::high_resolution_clock::now();
    // solve the problem
    pde_r0.solve();
    const auto t_sol = std::chrono::high_resolution_clock::now();
    std::cout << "DONE!\n";

    test.assembler_time_ = (std::chrono::duration_cast<std::chrono::milliseconds>(t_ass-t0)).count();
    test.solver_time_ = (std::chrono::duration_cast<std::chrono::microseconds>(t_sol-t1)).count();
    test.solution_ = pde_r0.solution();

    std::cout << "Computing error norms...\n";
    // to estimate our errors wrt geopde
    double val = 0.;
    double val_x = 0.;
    double val_y = 0.;
    double val_geo = 0.;
    double val_x_geo = 0.;
    double val_y_geo = 0.;
    double sum_L2 = 0.;
    double sum_Linf = 0.;
    double sum_H1 = 0.;
    double sum_L2_geo = 0.;
    double sum_Linf_geo = 0.;
    double sum_H1_geo = 0.;

    // read geopde solution
    SpMatrix<double> geo_sol;
    Eigen::loadMarket(geo_sol, test.solution_geopde_filename_);

    DMatrix<double> qn = pde_r0.quadrature_nodes();

    // needed to evaluate our solution on the physical domain
    auto F = mesh.parametrization();
    auto J = mesh.gradient();
    
    // for each quadrature node...
    for (std::size_t i = 0; i < qn.rows(); ++i){

        // current point
        auto p = qn.row(i);

        val = 0.;
        val_x = 0.;
        val_y = 0.;
        val_geo = 0.;
        val_x_geo = 0.;
        val_y_geo = 0.;

        for (std::size_t k = 0; k < mesh.basis().size(); k++){
            val += pde_r0.solution()(k) * mesh.basis()[k](p);
            val_x += pde_r0.solution()(k) * mesh.basis()[k].derive()[0](p);
            val_y += pde_r0.solution()(k) * mesh.basis()[k].derive()[1](p);
            val_geo += geo_sol.coeff(k, 0) * mesh.basis()[k](p);
            val_x_geo += geo_sol.coeff(k, 0) * mesh.basis()[k].derive()[0](p);
            val_y_geo += geo_sol.coeff(k, 0) * mesh.basis()[k].derive()[1](p);
        }

        auto x = F(p);
        auto Jx = J(p);
        // difference between our solution and geopde one in the current point
        double diff = (val - val_geo);
        double diff_dx = (val_x - val_x_geo);
        double diff_dy = (val_y - val_y_geo);

        // update on sums
        sum_L2 += diff * diff;
        sum_Linf = (std::abs(diff) > sum_Linf) ? std::abs(diff) : sum_Linf;
        sum_H1 += diff_dx * diff_dx + diff_dy * diff_dy;

        sum_L2_geo += val_geo*val_geo;
        sum_Linf_geo = (std::abs(val_geo) > sum_Linf_geo) ? std::abs(val_geo) : sum_Linf_geo;
        sum_H1_geo += val_x_geo*val_x_geo + val_y_geo*val_y_geo;

    }

    double L2_norm_geo = std::sqrt(sum_L2_geo);
    double Linf_norm_geo = sum_Linf_geo;
    double H1_norm_geo = std::sqrt(sum_H1_geo);
    test.err_L2_ = std::sqrt(sum_L2)/L2_norm_geo;
    test.err_L_inf_ = sum_Linf/Linf_norm_geo;
    test.err_H1_ = std::sqrt(sum_H1)/H1_norm_geo;
    std::cout << "DONE!\n";
    std::cout << "Results:\n";
    std::cout << "Test name: " << test.name_ << "\n";
    std::cout << "Refinement number: " << test.ref_n_ << "\n";
    std::cout << "Assembler time: " << test.assembler_time_ << "ms\n";
    std::cout << "Solver time: " << test.solver_time_ << "us\n";
    std::cout << "Error L2 norm: " << test.err_L2_ << "\n";
    std::cout << "Error Linf norm: " << test.err_L_inf_ << "\n";
    std::cout << "Error H1 norm: " << test.err_H1_ << "\n\n";

}

void post_processing (std::vector<testData> & all_tests, std::string  geopde_times_filename){
    // Write data
    std::cout << "Starting postprocessing...\n";
    SpMatrix<double> geo_times;
    Eigen::loadMarket(geo_times, geopde_times_filename);
    std::ofstream file_t("time_results.dat");
    std::ofstream file_err("error_results.dat");

    file_t<<"# ref_n\t t_a_fda\t t_a_geo\t t_s_fda\t t_s_geo\n";
    file_err<<"# ref_n\t err_l2_fda\t err_linf_fda\t err_H1_fda\n";
    for(unsigned int i = 0; i < all_tests.size(); ++i)
    {
        file_t << std::scientific<< all_tests[i].ref_n_ << "\t" << all_tests[i].assembler_time_ << "\t" 
            << geo_times.coeff(0,i)*1e3 << "\t" << all_tests[i].solver_time_ << "\t"
            << geo_times.coeff(1,i)*1e6  << "\t" << std::endl;
        file_err << std::scientific << all_tests[i].ref_n_ << "\t" << all_tests[i].err_L2_ << "\t" << all_tests[i].err_L_inf_
            << "\t" <<  all_tests[i].err_H1_<< "\t" << std::endl;
    }
    file_t.close();
    file_err.close();
    std::cout << "DONE!\n";
}