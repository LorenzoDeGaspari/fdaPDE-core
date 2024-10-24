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
#include <memory>

#include <fdaPDE/utils.h>
#include <fdaPDE/mesh.h>
using fdapde::core::Element;
using fdapde::core::NaiveSearch;
using fdapde::core::BarycentricWalk;
using fdapde::core::ADT;

#include "utils/mesh_loader.h"
using fdapde::testing::MESH_TYPE_LIST;
using fdapde::testing::MeshLoader;

// test fixture. ADT and bruteforce can work on this fixture (barycentric walk is not able to handle manifolds)
template <typename E> struct point_location_test : public ::testing::Test {
    MeshLoader<E> meshLoader {};   // use default mesh
    static constexpr unsigned int M = MeshLoader<E>::M;
    static constexpr unsigned int N = MeshLoader<E>::N;
};
TYPED_TEST_SUITE(point_location_test, MESH_TYPE_LIST);

TYPED_TEST(point_location_test, naive_search) {
    // build search engine
    NaiveSearch<TestFixture::M, TestFixture::N> engine(this->meshLoader.mesh);
    // build test set
    std::vector<std::pair<std::size_t, SVector<TestFixture::N>>> testSet = this->meshLoader.sample(100);
    // test all queries in test set
    std::size_t matches = 0;
    for (auto query : testSet) {
        auto e = engine.locate(query.second);
        if (e != nullptr && e->ID() == query.first) matches++;
    }
    EXPECT_EQ(matches, 100);
}

TYPED_TEST(point_location_test, alternating_digital_tree) {
    // build search engine
    ADT<TestFixture::M, TestFixture::N> engine(this->meshLoader.mesh);
    // build test set
    std::vector<std::pair<std::size_t, SVector<TestFixture::N>>> testSet = this->meshLoader.sample(100);
    // test all queries in test set
    std::size_t matches = 0;
    for (auto query : testSet) {
        auto e = engine.locate(query.second);
        if (e != nullptr && e->ID() == query.first) { matches++; }
    }
    EXPECT_EQ(matches, 100);
}

// barycentric walk cannot be applied to manifold mesh, filter out manifold cases at compile time
TYPED_TEST(point_location_test, barycentric_walk) {
    if constexpr (TestFixture::N == TestFixture::M) {
        BarycentricWalk<TestFixture::M, TestFixture::N> engine(this->meshLoader.mesh);
        // build test set
        std::vector<std::pair<std::size_t, SVector<TestFixture::N>>> testSet = this->meshLoader.sample(100);
        // test all queries in test set
        std::size_t matches = 0;
        for (auto query : testSet) {
            auto e = engine.locate(query.second);
            if (e != nullptr && e->ID() == query.first) matches++;
        }
        EXPECT_EQ(matches, 100);
    } else {
        // nothing to do in manifold cases here.
        SUCCEED();
    }
}
