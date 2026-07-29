// SPDX-License-Identifier: MIT
// Fixed reproduction transport for the CGAL optional-provider spike.

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/version.h>
#include <boost/version.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Vertex_base = CGAL::Triangulation_vertex_base_with_info_2<unsigned, Kernel>;
using Face_base = CGAL::Triangulation_face_base_2<Kernel>;
using Data_structure = CGAL::Triangulation_data_structure_2<Vertex_base, Face_base>;
using Delaunay = CGAL::Delaunay_triangulation_2<Kernel, Data_structure>;
using Point = Kernel::Point_2;

namespace {

constexpr const char* contract = "jacobian.cgal-delaunay-spike/v1";

std::array<unsigned, 3> canonical_face(Delaunay::Face_handle face) {
  std::array<unsigned, 3> ids = {
      face->vertex(0)->info(),
      face->vertex(1)->info(),
      face->vertex(2)->info(),
  };
  if (CGAL::orientation(face->vertex(0)->point(), face->vertex(1)->point(),
                        face->vertex(2)->point()) == CGAL::RIGHT_TURN) {
    std::swap(ids[1], ids[2]);
  }
  auto minimum = std::min_element(ids.begin(), ids.end());
  std::rotate(ids.begin(), minimum, ids.end());
  return ids;
}

int unique_reproduction() {
  using FT = Kernel::FT;
  const std::vector<std::pair<Point, unsigned>> sites = {
      {Point(FT(0), FT(0)), 0},
      {Point(FT(4), FT(0)), 1},
      {Point(FT(5), FT(3)), 2},
      {Point(FT(2), FT(5)), 3},
      {Point(FT(-1), FT(3)), 4},
      {Point(FT(3) / FT(2), FT(2)), 5},
  };
  Delaunay triangulation;
  triangulation.insert(sites.begin(), sites.end());
  if (!triangulation.is_valid() || triangulation.number_of_vertices() != sites.size()) {
    return 2;
  }

  std::vector<std::array<unsigned, 3>> triangles;
  std::set<std::pair<unsigned, unsigned>> edges;
  std::set<std::pair<unsigned, unsigned>> hull_edges;
  for (auto face = triangulation.finite_faces_begin();
       face != triangulation.finite_faces_end(); ++face) {
    auto ids = canonical_face(face);
    triangles.push_back(ids);
    for (int index = 0; index < 3; ++index) {
      auto first = ids[index];
      auto second = ids[(index + 1) % 3];
      edges.emplace(std::min(first, second), std::max(first, second));
    }
  }
  for (auto edge = triangulation.finite_edges_begin();
       edge != triangulation.finite_edges_end(); ++edge) {
    auto face = edge->first;
    auto index = edge->second;
    auto mirror = face->neighbor(index);
    if (!triangulation.is_infinite(face) && !triangulation.is_infinite(mirror)) {
      continue;
    }
    auto first = face->vertex((index + 1) % 3)->info();
    auto second = face->vertex((index + 2) % 3)->info();
    hull_edges.emplace(std::min(first, second), std::max(first, second));
  }
  std::sort(triangles.begin(), triangles.end());

  std::cout << "contract " << contract << '\n';
  std::cout << "cgal_version " << CGAL_VERSION_STR << '\n';
  std::cout << "kernel Exact_predicates_exact_constructions_kernel\n";
  std::cout << "semantics REQUIRE_UNIQUE\n";
  std::cout << "applicable true\n";
  std::cout << "valid true\n";
  std::cout << "site_count " << sites.size() << '\n';
  for (const auto& triangle : triangles) {
    std::cout << "triangle " << triangle[0] << ' ' << triangle[1] << ' '
              << triangle[2] << '\n';
  }
  for (const auto& edge : edges) {
    std::cout << "edge " << edge.first << ' ' << edge.second << '\n';
  }
  for (const auto& edge : hull_edges) {
    std::cout << "hull_edge " << edge.first << ' ' << edge.second << '\n';
  }
  return 0;
}

int cocircular_reproduction() {
  using FT = Kernel::FT;
  const std::array<Point, 4> sites = {
      Point(FT(0), FT(0)),
      Point(FT(2), FT(0)),
      Point(FT(2), FT(2)),
      Point(FT(0), FT(2)),
  };
  const auto side = CGAL::side_of_oriented_circle(
      sites[0], sites[1], sites[2], sites[3]);
  std::cout << "contract " << contract << '\n';
  std::cout << "cgal_version " << CGAL_VERSION_STR << '\n';
  std::cout << "kernel Exact_predicates_exact_constructions_kernel\n";
  std::cout << "semantics REQUIRE_UNIQUE\n";
  std::cout << "applicable false\n";
  std::cout << "reason "
            << (side == CGAL::ON_ORIENTED_BOUNDARY ? "COCIRCULAR" : "PROBE_ERROR")
            << '\n';
  return side == CGAL::ON_ORIENTED_BOUNDARY ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return 64;
  }
  const std::string mode = argv[1];
  if (mode == "--version") {
    std::cout << contract << " CGAL " << CGAL_VERSION_STR << '\n';
    std::cout << "compiler " << __VERSION__ << '\n';
    std::cout << "boost " << BOOST_LIB_VERSION << '\n';
    return 0;
  }
  if (mode == "--unique") {
    return unique_reproduction();
  }
  if (mode == "--cocircular") {
    return cocircular_reproduction();
  }
  return 64;
}
