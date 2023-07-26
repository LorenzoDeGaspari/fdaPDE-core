#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <memory>

#include "../../src/utils/symbols.h"
#include "../../src/mesh/element.h"
using fdapde::core::Element;
#include "../../src/linear_algebra/vector_space.h"
using fdapde::core::VectorSpace;

#include "utils/mesh_loader.h"
using fdapde::testing::MESH_TYPE_LIST;
using fdapde::testing::MeshLoader;
#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;
using fdapde::testing::MACHINE_EPSILON;
#include "utils/utils.h"
using fdapde::testing::almost_equal;

// test fixture
template <typename E>
struct ElementTest : public ::testing::Test {
  MeshLoader<E> meshLoader{}; // use default mesh
  static constexpr unsigned int M = MeshLoader<E>::M;
  static constexpr unsigned int N = MeshLoader<E>::N;
};
TYPED_TEST_SUITE(ElementTest, MESH_TYPE_LIST);

// check computation of barycentric coordinates is coherent with well known properties
TYPED_TEST(ElementTest, BarycentricCoordinates) {
  // generate a random element, a random point inside it and compute its barycentric coordinates
  auto e = this->meshLoader.generateRandomElement();
  SVector<TestFixture::N> p = this->meshLoader.generateRandomPoint(e);
  SVector<TestFixture::M + 1> q = e.to_barycentric_coords(p); // compute barycentric coordinates of point p

  // the barycentric coordinates of a point inside the element sums to 1
  EXPECT_TRUE( almost_equal(q.sum(), 1.0) );
  // the barycentric coordinates of a point inside the element are all positive
  EXPECT_TRUE((q.array() >= 0).all());

  // a point outside the element has at least one negative barycentric coordinate
  auto f = this->meshLoader.generateRandomElement();
  while(e.ID() == f.ID()) f = this->meshLoader.generateRandomElement();
  
  if constexpr(TestFixture::N == TestFixture::M){
    EXPECT_FALSE((e.to_barycentric_coords(f.mid_point()).array() > 0).all());
  }else{
    // for manifolds we have to consider if the point x belongs to the space spanned by the element
    VectorSpace<TestFixture::M, TestFixture::N> vs = e.spanned_space();
    EXPECT_FALSE(vs.distance(f.mid_point()) < DOUBLE_TOLERANCE && (e.to_barycentric_coords(f.mid_point()).array() > 0).all());
  }
  
  // the barycentric coordinates of the mid point of an element are all equal to (1+M)^{-1} (M is the dimension of the space
  // where the element belongs)
  SVector<TestFixture::M + 1> expected = SVector<TestFixture::M + 1>::Constant(1.0/(1+TestFixture::M));
  EXPECT_TRUE((e.to_barycentric_coords(e.mid_point()) - expected).norm() < DOUBLE_TOLERANCE);

  // a vertex has all its barycentric coordinates equal to 0 except a single one
  for(std::size_t i = 0; i < e.coords().size(); ++i){
    SVector<TestFixture::N> node = e.coords()[i];
    q = e.to_barycentric_coords(node);
    EXPECT_TRUE(((q.array()-1).abs() < DOUBLE_TOLERANCE).count() == 1 &&
		(q.array() < DOUBLE_TOLERANCE).count() == TestFixture::M);
  }
}

// test midpoint is correctly computed
TYPED_TEST(ElementTest, MidPoint) {
  // generate random element
  auto e = this->meshLoader.generateRandomElement();
  SVector<TestFixture::N> m = e.mid_point();
  SVector<TestFixture::M + 1> b = e.to_barycentric_coords(m);
  // the midpoint of an element is strictly inside it <-> its barycentric coordinates are all strictly positive
  EXPECT_TRUE((b.array() > 0).all());
  // the barycentric coordinates of the midpoint are all approximately equal (the midpoint is the center of mass)
  for(std::size_t i = 0; i < TestFixture::M; ++i){
    double x = b[i];
    for(std::size_t j = i+1; j < TestFixture::M + 1; ++j){
      double y = b[j];
      EXPECT_TRUE( almost_equal(x, y) );
    }
  }
}

// chcek .contains is able to correctly evaluate when a point is contained or not in an element
TYPED_TEST(ElementTest, CanAssertIfPointIsInside) {
  // draw some element at random
  auto e = this->meshLoader.generateRandomElement();
  // expect the mid point of the element is contained in the element itself
  EXPECT_TRUE(e.contains(e.mid_point()));
  
  // generate random points inside the element and check they are all contained into it
  for(std::size_t i = 0; i < 100; ++i){
    SVector<TestFixture::N> p = this->meshLoader.generateRandomPoint(e);
    EXPECT_TRUE(e.contains(p));
  }

  // expect the midpoint of a different element is not contained in e
  auto f = this->meshLoader.generateRandomElement();
  while(e.ID() == f.ID()) f = this->meshLoader.generateRandomElement();
  EXPECT_FALSE(e.contains(f.mid_point()));
}

TEST(ElementTest, ElementMeasureIsWithinMachineEpsilon){
  // load mesh
  MeshLoader<Mesh2D<>> CShaped("c_shaped");
  // consider element with ID 175 and check its measure equals expected one
  double measure = 0.0173913024287495;
  EXPECT_TRUE( almost_equal(CShaped.mesh.element(175).measure(), measure) );
}

// test barycentric matrix and its inverse is computed correctly
TEST(ElementTest, BarycentricMatrixAndItsInverseAreWithinMachineEpsilon){
  // load mesh
  MeshLoader<Mesh2D<>> CShaped("c_shaped");
  auto e = CShaped.mesh.element(175);
  
  // set expected barycentric matrix for element with ID 175
  Eigen::Matrix<double, 2,2> barycentric_matrix;
  barycentric_matrix <<
    0.1666666666666650, 0.0418368195713161,
    0.0345649283581886, 0.2173721491722987;
  Eigen::Matrix<double, 2,2> M = e.barycentric_matrix();
  // check equality within a tolerance of machine epsilon
  for(std::size_t i = 0; i < 2; ++i)
    for(std::size_t j = 0; j < 2; ++j)
      EXPECT_TRUE( almost_equal(barycentric_matrix(i,j), M(i,j)) );

  // set expected inverse of the barycentric matrix for element with ID 175
  Eigen::Matrix<double, 2,2> inv_barycentric_matrix;
  inv_barycentric_matrix <<
    6.2494499783110298, -1.2028086954015513,
   -0.9937417999542519,  4.7916671954122458;
  Eigen::Matrix<double, 2,2> invM = e.inv_barycentric_matrix();
  // check equality within a tolerance of machine epsilon
  for(std::size_t i = 0; i < 2; ++i)
    for(std::size_t j = 0; j < 2; ++j)
      EXPECT_TRUE( almost_equal(inv_barycentric_matrix(i,j), invM(i,j)) );
}
