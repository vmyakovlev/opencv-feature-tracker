#ifndef __FEATURE_GROUPER_DFS_VISITOR_
#define __FEATURE_GROUPER_DFS_VISITOR_
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>

namespace SaunierSayed{

    /** \class TrackManagerDFSVistor

      This class implements a DFS visitor that can find the connected components in the graph.
      It is to be used by the GetConnectedComponents() method in TrackManager.

      The algorithm works by visiting the graph vertices in a DFS search. This visitor will assign
      each vertex with a number : the component id.

      \todo Use concept checking to make sure that the graph has a property called activated for each vertex
    */
    template<typename ComponentIDMap>
    class TrackManagerDFSVsitor : public boost::default_dfs_visitor {
        typedef typename boost::property_traits < ComponentIDMap >::value_type T;
    public:
        TrackManagerDFSVsitor(ComponentIDMap component_map)
            :component_map_(component_map), current_component_id_(0)
        {}

        template < typename Vertex, typename Graph >
          void initialize_vertex(Vertex u, const Graph & g) const
        {
            // We first initialize the component id to be num_vertices(g)
            // This value is outside the domain of possible ids for
            // a vertex
            put(component_map_, u, num_vertices(g));
        }

        template < typename Vertex, typename Graph >
          void discover_vertex(Vertex u, const Graph & g) //const
        {
            // If this vertex has not been assigned into a connected component
            // before, then we will give it a new component id
            // Of course, we need to make sure that it is activated as well
            if (g[u].activated && get(component_map_, u) != num_vertices(g)){
                put(component_map_, u, current_component_id_);
                current_component_id_++;
            }

        }

        template < typename Edge, typename Graph >
          void tree_edge(Edge e, const Graph & g) const
        {
            assert(get(component_map_, target(e,g)) != num_vertices(g));
            if (g[target(e,g)].activated){
                put(component_map_, target(e,g),  get(component_map_, source(e,g)));
            }

        }

        T current_component_id_;
        ComponentIDMap component_map_;
    };
};


#endif
