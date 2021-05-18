/**************************************************************************************************************

Cluster analysis algorithm: Dendrogram


You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

**************************************************************************************************************/

#ifndef _DMeans_H_
#define _DMeans_H_

#include <cassert>
#include <vector>
#include <memory>
#include <list>
#include "MeansT.h"

// #define DEBUG_DMEANS

namespace Clustering {

    template <class DataItem, class DataSetIterator, class DataMetric, class DataAggregator>
        class DMeansT {
            public: 
                typedef std::vector<DataItem> DataSet;
                typedef std::vector<DataItem> CenterSet;


            protected:
                DataSetIterator ds_begin, ds_end;
                DataMetric metric;
                DataAggregator aggregator;

                typedef std::pair<int, int> Leave;
                std::vector<Leave> leaves;
                std::vector<int> replace_with;
                std::vector<unsigned int> num_leaves;
                std::vector<unsigned int> depth;

                typedef std::vector<double> DistLine;
                std::vector<DistLine> dmat;

                unsigned int    m_maximum_clusters;
                double          m_max_distance;
                ClusterSet	    m_clusters;
                CenterSet	    m_nodes;
                CenterSet       m_centers;
                size_t          m_dataset_size;


            public:
                DMeansT(DataSetIterator begin, DataSetIterator end, 
                        const DataMetric & metric, const DataAggregator & aggregator, const double max_distance) : 
                    ds_begin(begin), ds_end(end), 
                    metric(metric), aggregator(aggregator), 
                    m_max_distance(max_distance){}

                ~DMeansT(void) {}

                void process(void) {
                    double dmax = 0.0;
                    // First build the distance matrix
                    size_t i = 0;
                    for (DataSetIterator d_it = ds_begin;d_it<ds_end;d_it++, i++) {
                        num_leaves.push_back(1);
                        dmat.push_back(DistLine(i,0.0));
                        for (unsigned int j=0;j<i;j++) {
                            dmat[i][j] = metric(*(ds_begin + i), *(ds_begin + j));
                            dmax = std::max(dmax,dmat[i][j]);
                        }
                    }
                    m_dataset_size = i;
                    leaves.assign(m_dataset_size*2-1,Leave(-1,-1));
                    replace_with.assign(m_dataset_size*2-1,-1);
                    depth.assign(m_dataset_size*2-1,0);
                    for (i=0;i<m_dataset_size-1;i++) {
                        num_leaves.push_back(0);
                        dmat.push_back(DistLine(m_dataset_size + i));
                    }

                    // Now build the dendrogram
                    m_nodes.resize(m_dataset_size-1);
                    i = 0;
                    for (DataSetIterator d_it = ds_begin;d_it<ds_end;d_it++, i++) {
                        Leave l = find_closest_pair(m_dataset_size+i,dmax);
#ifdef DEBUG_DMEANS
                        printf("%d: pair (%d,%d)\n",(int)i,l.first,l.second);
#endif
                        assert(l.first>=0);
                        assert(l.second>=0);
                        leaves[m_dataset_size+i] = l;
                        num_leaves[m_dataset_size+i] = num_leaves[l.first]+num_leaves[l.second];
                        depth[m_dataset_size+i] = std::max(depth[l.first],depth[l.second])+1;
                        replace_with[l.first] = m_dataset_size+i;
                        replace_with[l.second] = m_dataset_size+i;
                        std::vector<DataItem> pairs;
                        Membership members;
                        pairs.push_back(select(l.first)); members.push_back(0);
                        pairs.push_back(select(l.second));members.push_back(1);
                        m_nodes[i] = aggregator(pairs.begin(),members.begin(),members.end());
                        dmat[i+m_dataset_size].resize(i+m_dataset_size);
                        for (unsigned int j=0;j<dmat[i+m_dataset_size].size();j++) {
                            dmat[i+m_dataset_size][j] = metric(m_nodes[i],select(j));
                            dmax = std::max(dmax,dmat[i+m_dataset_size][j]);
                        }
                        if (i >= m_dataset_size-2) {
                            break;
                        }
                    }
                    // Finally cut the dendrogram at m_max_distance to build clusters
                    {
                        m_centers.clear();
                        m_clusters.clear();
                        int i=m_dataset_size+m_nodes.size()-1;
                        std::vector<bool> done(leaves.size(),false);
                        while (i>=0) {
                            if (done[i]) {
                                // do nothing, this is part of a cluster already
                            } else if ((leaves[i].first<0)||(leaves[i].second<0)) {
                                // This is a leave. It must be a single data point
                                m_centers.push_back(*(ds_begin+i));
                                Membership mship(1,i); 
                                m_clusters.push_back(mship);
                                done[i] = true;
                            } else if (dmat[leaves[i].first][leaves[i].second]>m_max_distance) {
                                // do nothing, we've done it or we'll get to it later
                                done[i] = true;
                            } else {
                                // anything below this leave is part of a cluster
                                Membership mship; 
                                m_centers.push_back(m_nodes[i-m_dataset_size]);
                                std::list<unsigned int> topop;
                                topop.push_back(i);
                                while (!topop.empty()) {
                                    unsigned int l = topop.front();
                                    topop.pop_front();
                                    if ((leaves[l].first<0)||(leaves[l].second<0)) {
                                        mship.push_back(l);
                                    } else {
                                        topop.push_back(leaves[l].first);
                                        topop.push_back(leaves[l].second);
                                    }
                                    done[l] = true;
                                }
                                m_clusters.push_back(mship);
                            }
                            i--;
                        }
                    }

                }

                unsigned int GetNumClusters() const {
                    return m_clusters.size();
                }

                inline const ClusterSet & clusters() const {return m_clusters;}
                inline const CenterSet & centers() const {return m_centers;}

                inline void get_clusters(ClusterSet & output_clusters) const {
                    output_clusters = m_clusters;
                }

                inline void get_cluster_centers(CenterSet & output_centers) const {
                    output_centers = m_centers;
                }

            protected:
                const DataItem & select(unsigned int i) const{
                    if (i < m_dataset_size) {
                        return *(ds_begin + i);
                    } else {
                        return m_nodes[i-m_dataset_size];
                    }
                }

                Leave find_closest_pair(unsigned int last_elem, double dmax) const {
                    double d = dmax+1;
                    Leave l(-1,-1);
                    for(unsigned int i=0;i<last_elem;i++) {
                        if (replace_with[i]>=0) continue;
                        for (unsigned int j=0;j<i;j++) {
                            if (replace_with[j]>=0) {
                                continue;
                            } 
                            if (dmat[i][j] < d) {
                                d = dmat[i][j];
                                l = Leave(i,j);
                            }
                        }
                    }
                    if (l.first < l.second) {
                        return Leave(l.second,l.first);
                    }
                    return l;
                }

                void build_label_list(std::vector<unsigned int> & labels,
                        std::vector<float> & node_pos,
                        std::vector<unsigned int> & inv_labels) const {
                    labels.clear();
                    node_pos.assign(leaves.size(),-1.0);
                    inv_labels.assign(leaves.size(),0);
                    std::list<unsigned int> topop;
                    topop.push_back(leaves.size()-1);
                    while (!topop.empty()) {
                        unsigned int l = topop.front();
                        topop.pop_front();
                        if ((leaves[l].first>=0)&&(leaves[l].second>=0)) {
                            topop.push_front(leaves[l].first);
                            topop.push_front(leaves[l].second);
                        } else {
                            inv_labels[l] = labels.size();
                            labels.push_back(l);
                        }
                    }
                    for (unsigned int i=0;i<m_dataset_size;i++) {
                        node_pos[i] = inv_labels[i];
                    }
                    for (unsigned int i=m_dataset_size;i<leaves.size();i++) {
                        node_pos[i] = (node_pos[leaves[i].first]*num_leaves[leaves[i].first]
                                +node_pos[leaves[i].second]*num_leaves[leaves[i].second]) / num_leaves[i];
                    }
                }

        };
};

#endif
