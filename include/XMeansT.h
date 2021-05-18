/**************************************************************************************************************

Cluster analysis algorithm: X-Means

Based on article description:
 - D.Pelleg, A.Moore. X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 2000.

Copyright (C) 2015    Andrei Novikov (pyclustering@yandex.ru)

pyclustering is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyclustering is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

**************************************************************************************************************/

#ifndef _XMeans_Templated_H_
#define _XMeans_Templated_H_

#include <assert.h>
#include <vector>
#include <random>

// #define DEBUG_XMEANS_T
#ifdef DEBUG_XMEANS_T
#include <fstream>
#endif

#include "MeansT.h"

namespace Clustering {


    template <class DataItem, class DataSetIterator, class DataMetric, class DataAggregator, class CenterSplitter>
        class XMeansT {
            public: 
                typedef std::vector<DataItem> CenterSet;

            protected:
                DataSetIterator ds_begin, ds_end;
                DataMetric metric;
                DataAggregator aggregator;
                CenterSplitter splitter;

                size_t    m_dimension;
                size_t    m_maximum_clusters;
                double          m_tolerance;
                double          m_split_ratio;
                ClusterSet	    m_clusters;
                CenterSet       m_centers;

                std::default_random_engine rge;
            public:
                XMeansT(DataSetIterator begin, DataSetIterator end, 
                        const CenterSet & initial_centers, 
                        size_t dimension,
                        const DataMetric & metric, 
                        const DataAggregator & aggregator, 
                        const CenterSplitter & splitter, 
                        const size_t kmax, const double minimum_change) :
                    ds_begin(begin), ds_end(end), 
                    metric(metric), aggregator(aggregator), splitter(splitter),
                    m_dimension(dimension), m_maximum_clusters(kmax), 
                    m_tolerance(minimum_change), m_split_ratio(0.25), 
                    m_clusters(initial_centers.size(),Membership()),
                    m_centers(initial_centers) {}

                ~XMeansT(void) {}

                size_t GetNumClusters() const {return m_clusters.size();}
                inline const ClusterSet & clusters() const {return m_clusters;}
                inline const CenterSet & centers() const {return m_centers;}

            public:
                void update_clusters(ClusterSet & clusters, const CenterSet & centers, const std::vector<size_t> & indices = std::vector<size_t>()) {
                    clusters.clear();
                    clusters.assign(centers.size(), Membership());

                    if (indices.empty()) {
                        size_t index_object = 0;
                        for (DataSetIterator it=ds_begin;it!=ds_end;it++,index_object++) {
                            size_t index_cluster = find_proper_cluster(centers, *it);
                            clusters[index_cluster].push_back(index_object);
                        }
                    } else {
                        for (std::vector<size_t>::const_iterator index_object = indices.begin(); index_object != indices.end(); index_object++) {
                            size_t index_cluster = find_proper_cluster(centers, *(ds_begin + *index_object));
                            clusters[index_cluster].push_back(*index_object);
                        }
                    }
                }

                void remove_empty_clusters(ClusterSet & clusters, CenterSet & centers) {
                    CenterSet new_centers;
                    ClusterSet new_clusters;
                    for (size_t i=0;i<clusters.size();i++) {
                        if (clusters[i].size() > 0) {
                            new_centers.push_back(centers[i]);
                            new_clusters.push_back(clusters[i]);
                        }
                    }
                    clusters = new_clusters;
                    centers = new_centers;
                }

                double update_centers(const ClusterSet & clusters, CenterSet & centers) {
                    double maximum_change = 0;

                    /* for each cluster */
                    assert(centers.size() == clusters.size());
                    for (size_t index_cluster = 0; index_cluster < clusters.size(); index_cluster++) {

                        DataItem total = aggregator(ds_begin,
                                clusters[index_cluster].begin(),
                                clusters[index_cluster].end());

                        double d = metric( centers[index_cluster], total);

                        if (d > maximum_change) {
                            maximum_change = d;
                        }

#ifdef DEBUG_XMEANS_T
                        std::cout << "Update centers: " << index_cluster << ": " << centers[index_cluster] << " -> " << total << " change " << d << "/" << maximum_change << std::endl;
#endif
                        centers[index_cluster] = total;
                    }
                    return maximum_change;
                }

                void improve_structure() {
                    CenterSet allocated_centers;
                    assert(m_centers.size() == m_clusters.size());

                    for (size_t index = 0; index < m_clusters.size(); index++) {
                        double sigma  = 0;
                        for (MemberSetIterator index_object = m_clusters[index].begin(); index_object != m_clusters[index].end(); index_object++) {
                            sigma += metric(*(ds_begin + *index_object), m_centers[index]);
                        }
                        sigma /= m_clusters[index].size();

#ifdef DEBUG_XMEANS_T
                        std::ofstream ocenters ("center", std::ofstream::out);
                        std::ofstream odata ("data", std::ofstream::out);

                        std::cout << "Refining cluster "<< index <<
                            " with " << m_clusters[index].size() << " elements" << std::endl;
#endif
                        if (m_clusters[index].size() < 2) {
                            allocated_centers.push_back(m_centers[index]);
#ifdef DEBUG_XMEANS_T
                            std::cout << "Kept" << std::endl;
#endif
                            continue;
                        }
#ifdef DEBUG_XMEANS_T
                        for (MemberSetIterator mb_it = m_clusters[index].begin();
                                mb_it != m_clusters[index].end(); mb_it++) {
                            std::cout << *mb_it << " ";
                        }
                        std::cout << std::endl;
#endif
                        std::pair<DataItem,DataItem> children = splitter(m_centers[index], m_split_ratio*sigma, rge);
                        CenterSet parent_child_centers;
                        parent_child_centers.push_back(children.first);
                        parent_child_centers.push_back(children.second);

                        /* solve k-means problem for children where data of parent are used */
                        ClusterSet parent_child_clusters(2,Membership());

                        improve_parameters(parent_child_clusters, parent_child_centers, m_clusters[index]);
#ifdef DEBUG_XMEANS_T
                        ocenters << m_centers[index] << " 0" << std::endl;
                        ocenters << parent_child_centers[0] << " 1" << std::endl;
                        ocenters << parent_child_centers[1] << " 2" << std::endl;

                        for (size_t j=0;j<parent_child_clusters.size();j++) {
                            std::cout << j << ": ";
                            for (size_t k=0;k<parent_child_clusters[j].size();k++) {
                                std::cout << parent_child_clusters[j][k] << " ";
                                odata << *(ds_begin + parent_child_clusters[j][k]) << " " << j << std::endl;
                            }
                            std::cout << std::endl;
                        }

                        odata.close();
                        ocenters.close();

#endif

                        /* splitting criterion */
                        ClusterSet parent_cluster(1, m_clusters[index]);
                        CenterSet parent_center(1, m_centers[index]);

                        double parent_scores = splitting_criterion(parent_cluster, parent_center);
                        double child_scores = splitting_criterion(parent_child_clusters, parent_child_centers);
#ifdef DEBUG_XMEANS_T
                        std::cout << "Improvement: Cluster " << index << 
                            " Score " << parent_scores << " Children " << child_scores << std::endl;
#endif


                        /* take the best representation of the considered data */
                        if (std::isnan(child_scores) || (parent_scores >= child_scores)) {
                            allocated_centers.push_back(m_centers[index]);
#ifdef DEBUG_XMEANS_T
                            std::cout << "Kept" << std::endl;
#endif
                        } else {
                            allocated_centers.push_back(parent_child_centers[0]);
                            allocated_centers.push_back(parent_child_centers[1]);
#ifdef DEBUG_XMEANS_T
                            std::cout << "Split" << std::endl;
#endif
                        }
#ifdef DEBUG_XMEANS_T
                        getchar();
#endif
                    }

                    /* update current centers */
                    m_centers = allocated_centers;
                }

                void improve_parameters(ClusterSet & clusters, CenterSet & centers, const std::vector<size_t> & indices=std::vector<size_t>()) {
                    // K-Mean
                    double current_change = std::numeric_limits<double>::max();
                    while(current_change > m_tolerance) {
                        update_clusters(clusters, centers, indices);
                        current_change = update_centers(clusters, centers);
#ifdef DEBUG_XMEANS_T
                        std::cout << "Update centers: current change = " << current_change << " tol " << m_tolerance << std::endl;
#endif
                    }
                }


                double splitting_criterion(const ClusterSet & clusters, const CenterSet & centers) const {

                    double sigma = 0.0;
                    size_t K = centers.size();
                    size_t N = 0;

                    for (size_t index_cluster = 0; index_cluster < clusters.size(); index_cluster++) {
                        for (MemberSetIterator index_object = clusters[index_cluster].begin(); index_object != clusters[index_cluster].end(); index_object++) {
                            sigma += metric(*(ds_begin + *index_object), centers[index_cluster]);
                        }

                        N += clusters[index_cluster].size();
                    }

                    sigma /= (double) (N - K);

                    /* splitting criterion */
                    double score = 0;
                    for (size_t index_cluster = 0; index_cluster < centers.size(); index_cluster++) {
                        double n = (double) clusters[index_cluster].size();
                        // BIC
                        score += n * std::log(n) - n * std::log(N) - n * std::log(2.0 * M_PI) / 2.0 - n * m_dimension * std::log(sigma) / 2.0 - (n - K) / 2.0;
                    }

                    return score;
                }

                size_t find_proper_cluster(const CenterSet & centers, const DataItem & di) const {
                    size_t index_optimum = 0;
                    double distance_optimum = std::numeric_limits<double>::max();

                    for (size_t index_cluster = 0; index_cluster < centers.size(); index_cluster++) {
                        double d = metric( di, centers[index_cluster]);
                        if (d < distance_optimum) {
                            index_optimum = index_cluster;
                            distance_optimum = d;
                        }
                    }
                    return index_optimum;
                }

            public:
                void process(void) {
                    size_t current_number_clusters = m_clusters.size();

                    while (current_number_clusters < m_maximum_clusters) {
                        improve_parameters(m_clusters, m_centers);
                        size_t old_k = m_centers.size();
                        improve_structure();
                        current_number_clusters = m_centers.size();
#ifdef DEBUG_XMEANS_T
                        std::cout << "Cluster change: " << old_k << " -> " << current_number_clusters << std::endl;
#endif
                        if (current_number_clusters == old_k) {
                            break;
                        }
                    }
                    // Final refinement to eliminate empty clusters
                    remove_empty_clusters(m_clusters, m_centers);
                    update_centers(m_clusters, m_centers);
                }


        };

};
#endif
