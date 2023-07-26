#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <limits>
#include <string>
#include <type_traits>

#include "../../src/utils/symbols.h"
#include "../../src/utils/compile_time.h"
using fdapde::core::ct_binomial_coefficient;
#include "../../src/finite_elements/basis/lagrangian_basis.h"
#include "../../src/finite_elements/basis/multivariate_polynomial.h"
using fdapde::core::LagrangianBasis;
using fdapde::core::MultivariatePolynomial;
#include "../../src/mesh/reference_element.h"
using fdapde::core::point_list;
using fdapde::core::ReferenceElement;
#include "../../src/finite_elements/integration/integrator.h"
using fdapde::core::Integrator;
#include "../../src/fields/vector_field.h"
#include "../../src/fields/matrix_field.h"
#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;
#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
using fdapde::testing::MESH_TYPE_LIST;

// a type representing a compile time pair of integer values
template <int i, int j> struct int_pair {
  static constexpr int first  = std::integral_constant<int, i>::value;
  static constexpr int second = std::integral_constant<int, j>::value;
};

template <typename E>
class LagrangianBasisTest : public ::testing::Test {
public:
  static constexpr int N = E::first;
  static constexpr int R = E::second;
  static constexpr std::size_t n_basis = ct_binomial_coefficient(N+R,R);

  // constructor
  LagrangianBasisTest() = default;
  
  // transforms the i-th point in a point_list to an SVector
  template <long unsigned int M, long unsigned int K>
  SVector<M> toSVector(const point_list<M, K> &pList, std::size_t i) const {
    SVector<M> result{};
    for(std::size_t j = 0; j < M; ++j){
      result[j] = pList[i][j];
    }
    return result;
  };
};

// pair format: <x,y> : x dimension of space, y order of mesh
using pairs = ::testing::Types<int_pair<1,1>, int_pair<2,1>, int_pair<3,1>   // order 1 elements (linear finite elements)
			      ,int_pair<1,2>, int_pair<2,2>, int_pair<3,2>>; // order 2 elements (quadratic finite elements)
TYPED_TEST_SUITE(LagrangianBasisTest, pairs);

// tests a Lagrangian basis can be successfully built over the reference unit simplex
TYPED_TEST(LagrangianBasisTest, DefineOverReferenceElement) {
  // create lagrangian basis over unit dimensional simplex
  LagrangianBasis<TestFixture::N, TestFixture::R> basis{};

  // expect correct number of basis functions
  EXPECT_EQ(basis.size(), TestFixture::n_basis);
  // check lagrangian property (each basis function is 1 in one and only one node and 0 in any other)
  for(const MultivariatePolynomial<TestFixture::N, TestFixture::R>& b : basis){
    std::size_t num_ones = 0, num_zeros = 0;
    for(std::size_t i = 0; i < TestFixture::n_basis; ++i){ // there are as many nodes as basis functions
      SVector<TestFixture::N> p = this->toSVector(ReferenceElement<TestFixture::N, TestFixture::R>::nodes, i);
      if(std::abs(b(p) - 1.0) < DOUBLE_TOLERANCE){
	num_ones++;
      }else if(b(p) < DOUBLE_TOLERANCE){
	num_zeros++;
      }
    }
    EXPECT_EQ(num_ones + num_zeros, TestFixture::n_basis);
    EXPECT_EQ(num_ones, 1);
    EXPECT_EQ(num_zeros, TestFixture::n_basis - 1);
  }
}

// test linear elements behave correctly on reference element
TEST(LagrangianBasisTest, LinearReferenceElement) {
  // create finite linear elements over unit reference simplex
  LagrangianBasis<2,1> basis{};
  SVector<2> p(0,0); // define evaluation point

  // basis functions are defined in counterclockwise order starting from node (0,0)
  // for linear elements we get
  // (0,0) -> \Nabla phi_{0} = [-1 -1]
  // (1,0) -> \Nabla phi_{0} = [ 1  0]
  // (0,1) -> \Nabla phi_{0} = [ 0  1]
  std::vector<SVector<2>> gradients({SVector<2>(-1.0,-1.0), SVector<2>(1.0, 0.0), SVector<2>(0.0, 1.0)});

  // check gradient of each basis function equals the expected one
  for(std::size_t i = 0; i < basis.size(); ++i){
    // extract gradient of basis function
    VectorField<2> grad = basis[i].derive();
    for(std::size_t j = 0; j < 2; ++j)
      EXPECT_TRUE( almost_equal(grad(p)[j], gradients[i][j]) );
  }
}

// test quadratic elements behave correctly on reference element
TEST(LagrangianBasisTest, QuadraticReferenceElement) {
  // create finite linear elements over unit reference simplex
  LagrangianBasis<2,2> basis{};
  SVector<2> p(0.5, 0.5); // define evaluation point

  // basis functions are defined following the enumeration:
  // 1 -> (0,0), 2 -> (1,0), 3 -> (0,1), 4 -> (0.5, 0), 5-> (0, 0.5), 6 -> (0.5, 0.5)

  // expected gradient of basis functions in p using analytical expression
  SMatrix<6,2> gradients;
  gradients <<
    1-4*(1-p[0]-p[1]), 1-4*(1-p[0]-p[1]), // \nabla \psi_1
    4*p[0]-1, 0,                          // \nabla \psi_2
    0,  4*p[1]-1,                         // \nabla \psi_3
    4*(1-2*p[0]-p[1]), -4*p[0],           // \nabla \psi_4
    -4*p[1], 4*(1-p[0]-2*p[1]),           // \nabla \psi_5
    4*p[1], 4*p[0];                       // \nabla \psi_6

  // check gradient of each basis function equals the expected one
  for(std::size_t i = 0; i < basis.size(); ++i){
    // extract gradient of basis function
    VectorField<2> grad = basis[i].derive();
    for(std::size_t j = 0; j < 2; ++j)
      EXPECT_TRUE( almost_equal(grad(p)[j], gradients(i,j)) );
  }
}

// test linear elements behave correctly on generic mesh elements
TEST(LagrangianBasisTest, LinearPhysicalElement){
  // load sample mesh
  MeshLoader<Mesh2D<>> CShaped("c_shaped");
  auto e = CShaped.mesh.element(175); // reference element for this test
  // get quadrature nodes over the mesh to define an evaluation point
  IntegratorTable<2,6> integrator{};
  SVector<2> p; p << integrator.nodes[0][0], integrator.nodes[0][1]; // define evaluation point
  // define vector of expected gradients, in the order with which basis are looped
  std::vector<SVector<2>> gradients({
      SVector<2>(-5.2557081783567776, -3.5888585000106943),
      SVector<2>( 6.2494499783110298, -1.2028086954015513),
      SVector<2>(-0.9937417999542519,  4.7916671954122458)
    });
  // use the barycentric matrix of e and the basis defined over the reference element
  Eigen::Matrix<double, 2, 2> invJ = e.inv_barycentric_matrix().transpose();
  LagrangianBasis<2,1> refBasis{};

  for(std::size_t i = 0; i < refBasis.size(); ++i){
    VectorField<2, 2> grad = invJ * refBasis[i].derive();
    for(std::size_t j = 0; j < 2; ++j)
      EXPECT_TRUE( almost_equal(grad(p)[j], gradients[i][j]) );
  }
}

// test quadratic elements behave correctly on generic mesh elements
TEST(LagrangianBasisTest, QuadraticPhysicalElement){
  // load sample mesh, Prepare the mesh to accept an order 2 basis
  MeshLoader<Mesh2D<2>> CShaped("c_shaped");
  auto e = CShaped.mesh.element(175); // reference element for this test
  // get quadrature nodes over the mesh to define an evaluation point
  IntegratorTable<2,6> integrator{};
  SVector<2> p; p << integrator.nodes[0][0], integrator.nodes[0][1]; // define evaluation point
  
  // define a linear finite element basis over e
  // define vector of expected gradients at quadrature node, in the order with which basis are looped
  std::vector<SVector<2>> gradients({
      SVector<2>( 2.9830765115928704,  2.0369927574935405),
      SVector<2>( 4.8982811692194259, -0.9427541948981384),
      SVector<2>(-0.7788888242446018,  3.7556798236502558),
      SVector<2>(-6.6727629051483941, -6.9218931297696376),
      SVector<2>(-9.8048064747509027, -4.3298093852388320),
      SVector<2>( 9.3751005233316018,  6.4017841287628112)
    });
  // use the barycentric matrix of e and the basis defined over the reference element
  Eigen::Matrix<double, 2, 2> invJ = e.inv_barycentric_matrix().transpose();  
  LagrangianBasis<2,2> refBasis{};

  for(std::size_t i = 0; i < refBasis.size(); ++i){
    VectorField<2, 2> grad = invJ * refBasis[i].derive();
    for(std::size_t j = 0; j < 2; ++j)
      EXPECT_TRUE( almost_equal(grad(p)[j], gradients[i][j]) );
  }
}
