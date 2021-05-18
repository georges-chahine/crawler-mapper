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

#ifndef _Means_Templated_H_
#define _Means_Templated_H_

#include <assert.h>
#include <vector>
#include <random>

// #define DEBUG_XMEANS_T
#ifdef DEBUG_XMEANS_T
#include <fstream>
#endif

namespace Clustering {
    typedef std::vector<size_t> Membership;
    typedef std::vector<Membership> ClusterSet;
    typedef Membership::const_iterator MemberSetIterator;

    // Functors
    // DataMetric: distance between two data item
    //      double operator()(const DataItem & d1, const DataItem & d2) const
    // DataAggregator: average of a cluster 
    //      DataItem operator()(DataSetIterator begin, MemberIterator members_begin, MemberIterator members_end) const;
    // CenterSplitter: randomly separate a center into two very close center
    //      std::pair<DataItem,DataItem> operator()(const DataItem & d, std::random_engine & rge) const

    template <class NumberType>
        struct DataMetricFP {
            double operator()(const NumberType & a, const NumberType & b) const {
                return fabs(a-b);
            }
        };


    template <class NumberType, class DataSetIterator>
        struct DataAggregatorFP {
            NumberType operator()(DataSetIterator ds_begin, MemberSetIterator mb_begin, MemberSetIterator mb_end) const {
                size_t count = 0;
                NumberType s = 0;
                for (MemberSetIterator mb_it = mb_begin; mb_it != mb_end; mb_it++) {
                    s += *(ds_begin + *mb_it);
                    count += 1;
                }
                return s / count;
            }
        };

    template <class NumberType, class Generator=std::default_random_engine> 
        class CenterSplitterFP {
            public:
                CenterSplitterFP() {}
                std::pair<NumberType,NumberType> operator()(const NumberType & c, double scale, Generator & rge) const {
                    return std::pair<NumberType,NumberType>(c-scale, c+scale);
                }

        };
    typedef CenterSplitterFP<float> CenterSplitterFloat;
    typedef CenterSplitterFP<double> CenterSplitterDouble;




};
#endif
