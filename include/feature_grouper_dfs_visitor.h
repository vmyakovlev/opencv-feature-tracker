/*///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//*/
#ifndef __FEATURE_GROUPER_DFS_VISITOR_
#define __FEATURE_GROUPER_DFS_VISITOR_
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <stdlib.h>
namespace SaunierSayed{

    /** \class TrackManagerDFSVistor

      This class implements a DFS visitor that can find the connected components in the graph.
      It is to be used by the GetConnectedComponents() method in TrackManager.

      The algorithm works by visiting the graph vertices in a DFS search. This visitor will assign
      each vertex with a number : the component id. A vertex is given a valid component id ONLY if its
      activated property is true. If not, this vertex will receive a component id >= num_components.

      For example, we have 3 nodes, the first two nodes are activated: they might have component ids 0 and 1.
      The third node is not activated, it will get the component id 3 (3 is outside of the range of possible component
      ids)

      \todo Use concept checking to make sure that the graph has a property called activated for each vertex
    */
    template<typename ComponentIDMap>
    class TrackManagerDFSVsitor : public boost::default_dfs_visitor {
        typedef typename boost::property_traits < ComponentIDMap >::value_type T;
    public:
        TrackManagerDFSVsitor(ComponentIDMap component_map, T & num_components)
            :component_map_(component_map), current_component_id_(num_components)
        {
            current_component_id_ = 0;
        }

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
            if (g[u].activated && get(component_map_, u) == num_vertices(g)){
                put(component_map_, u, current_component_id_);
                current_component_id_++;
            }

            // debug
            // printf("Visited %d: active? %d. Component id: %ld/%ld\n", g[u].id, g[u].activated, get(component_map_, u), num_vertices(g));
        }

        template < typename Edge, typename Graph >
          void tree_edge(Edge e, const Graph & g) const
        {
            // sanity check
            //assert(get(component_map_, source(e,g)) != num_vertices(g));
            // NOTE: the above cannot be guaranteed, think of a node with two children.
            //assert(get(component_map_, target(e,g)) != num_vertices(g));
            // NOTE: the above cannot be guaranteed, think of three vertices connected to each other

            // if this adjacent vertex is activated, assign it the same component id
            if (g[target(e,g)].activated){
                put(component_map_, target(e,g),  get(component_map_, source(e,g)));
            }

        }

        T & current_component_id_;
        ComponentIDMap component_map_;
    };
};


#endif
