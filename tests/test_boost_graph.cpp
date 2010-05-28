#include <gtest/gtest.h>
#include <vector>
#include <cv.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include "SaunierSayed_feature_grouping.h"

using namespace boost;

typedef adjacency_list<vecS, vecS, bidirectionalS, no_property, SaunierSayed::LinkInformation> Graph;
// convenient usage instead of integers
enum { A, B, C, D, E, F, G };

class BoostGraphTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        int num_vertices = 5;

        // writing out the edges in the graph
        typedef std::pair<int, int> Edge;
        Edge edge_array[] =
        { Edge(A,B), Edge(A,D), Edge(C,A), Edge(D,C),
          Edge(C,E), Edge(B,D), Edge(D,E) };
        const int num_edges = sizeof(edge_array)/sizeof(edge_array[0]);

        // Using edge iterator constructor (more efficient than calling add_edge)
        g = Graph(edge_array, edge_array + num_edges, num_vertices);

    }

    // virtual void TearDown() {}
    Graph g;
};

TEST_F(BoostGraphTest, SimpleAdditionRemovalOfVertices){
    ASSERT_EQ(5, num_vertices(g));

    Graph::vertex_descriptor u1,u2;
    std::pair<Graph::edge_descriptor, bool> v;

    // add a  new vertex
    u1 = add_vertex(g);
    u2 = add_vertex(g);

    // add an edge between these vertices
    v = add_edge(u1,u2,g);

    // stack on the properties too
    g[v.first].min_distance = 3.5;
    g[v.first].max_distance = 5;

    ASSERT_TRUE(v.second);

    ASSERT_EQ(7, num_vertices(g));
    ASSERT_EQ(8, num_edges(g));

    // severe the edge
    remove_edge(u1,u2,g);
    ASSERT_EQ(7, num_edges(g));
}

TEST_F(BoostGraphTest, StoringVertexDescriptors){
    std::vector<Graph::vertex_descriptor> vertex_descriptors;

    Graph::vertex_iterator vi, viend;

    for (tie(vi,viend) = vertices(g); vi != viend; ++vi){
        vertex_descriptors.push_back(*vi);
    }

    vertex_descriptors.push_back(add_vertex(g));

    ASSERT_EQ(num_vertices(g), vertex_descriptors.size());

    // NOTE: if we are to use vecS as the storage for vertices or edges, remove_vertex will invalidate all descriptors
    //       consider using listS instead. For other options and more information see: http://www.boost.org/doc/libs/1_35_0/libs/graph/doc/adjacency_list.html
    //       , search for remove_vertex(
}

TEST_F(BoostGraphTest, EdgeTraversal){
    //std::pair<edge_iterator, edge_iterator> edge_it = edges(g);

    Graph::edge_iterator vi, viend;
    int num_edge_traversed = 0;
    for (boost::tie(vi, viend) = edges(g); vi != viend; ++vi) {
        std::cout << "Edge " << source(*vi, g) << " => " << target(*vi, g) << ": " << g[*vi].min_distance << " " << g[*vi].max_distance << std::endl;
        num_edge_traversed++;
    }

    ASSERT_EQ(num_edges(g), num_edge_traversed);
}

TEST_F(BoostGraphTest, EdgeTraversalOnlyFromInterestVertices){
    int interested_vertices_id[] = {A,B,E};

    Graph::vertex_descriptor v;
    for (int i=0; i<3; i++){
        v = vertex(interested_vertices_id[i],g);

        ASSERT_EQ(interested_vertices_id[i], (int)v);
    }
}
