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

#ifndef _XMeans_H_
#define _XMeans_H_

#include <vector>

namespace Clustering {

    class XMeans {
        public: 
            typedef std::vector<unsigned int> Membership;
            typedef std::vector<Membership> ClusterSet;
            typedef std::vector<double> DataItem;
            typedef std::vector<DataItem> DataSet;
            typedef std::vector<DataItem> CenterSet;

            typedef double (*NormFunction)(const DataItem * const point1, const DataItem * const point2) ;


            typedef enum {EUCLIDEAN_SQR, EUCLIDEAN_EXP, EUCLIDEAN, NORM_L1, NORM_LInf, NORM_MANUAL} NormType;
        protected:
            const DataSet & m_dataset;

            unsigned int    m_maximum_clusters;
            double          m_tolerance;
            ClusterSet	    m_clusters;
            CenterSet       m_centers;
            NormType        m_norm;
            DataItem        m_weights;

            NormFunction manual_norm;

        public:
            XMeans(const DataSet & data, const CenterSet & initial_centers, 
                    const unsigned int kmax, const double minimum_change, 
                    const NormType norm=EUCLIDEAN_SQR);

            XMeans(const DataSet & data, const CenterSet & initial_centers, 
                    const unsigned int kmax, const double minimum_change, 
                    const DataItem & weights, const NormType norm=EUCLIDEAN_SQR);

            ~XMeans(void);

            void process(void);

            void setManualDistance(NormFunction f) {
                manual_norm = f;
            }

            unsigned int GetNumClusters() const;

            inline const ClusterSet & clusters() const {return m_clusters;}
            inline const CenterSet & centers() const {return m_centers;}

            inline void get_clusters(ClusterSet & output_clusters) const {
                output_clusters = m_clusters;
            }

            inline void get_cluster_centers(CenterSet & output_centers) const {
                output_centers = m_centers;
            }

            static double euclidean_distance_exp(const DataItem * const point1, const DataItem * const point2) ;
            static double euclidean_distance_sqr(const DataItem * const point1, const DataItem * const point2) ;
            static double euclidean_distance(const DataItem * const point1, const DataItem * const point2) ;
            static double l1_distance(const DataItem * const point1, const DataItem * const point2) ;
            static double linf_distance(const DataItem * const point1, const DataItem * const point2) ;
            double distance(const DataItem * const point1, const DataItem * const point2) const;

        protected:
            void update_clusters(ClusterSet & clusters, const CenterSet & centers, const std::vector<unsigned int> & available_indexes);

            double update_centers(const ClusterSet & clusters, CenterSet & centers);

            void improve_structure(void);

            void improve_parameters(ClusterSet & clusters, CenterSet & centers, const std::vector<unsigned int> & available_indexes);

            double splitting_criterion(const ClusterSet & clusters, const CenterSet & centers) const;

            unsigned int find_proper_cluster(const CenterSet & analysed_centers, const std::vector<double> & point) const;
    };
};

#endif
