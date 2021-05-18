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

namespace Clustering {

    class DMeans {
        public: 
            typedef std::vector<unsigned int> Membership;
            typedef std::vector<Membership> ClusterSet;
            typedef std::vector<double> DataItem;
            typedef std::vector<DataItem> DataSet;
            typedef std::vector<DataItem> CenterSet;

            typedef double (*NormFunction)(const DataItem & point1, const DataItem & point2) ;

            struct Averager {
                template <class Iterator>
                    bool operator()(Iterator start, Iterator end, DataItem & result) {
                        result.assign(result.size(),0.0);
                        unsigned int count = 0;
                        for (Iterator it=start;it!=end;it++) {
                            for (unsigned int i=0;i<result.size();i++) {
                                result[i] += (*it)[i];
                            }
                        }
                        if (count > 0) {
                            for (unsigned int i=0;i<result.size();i++) {
                                result[i] /= count;
                            }
                        }
                        return true;
                    }
                bool operator()(const DataItem & p1, double w1, const DataItem & p2, double w2, DataItem & result) {
                    assert(p1.size()==p2.size());
                    result.resize(p1.size());
                    for (unsigned int i=0;i<result.size();i++) {
                        result[i] = (p1[i]*w1+p2[i]*w2)/(w1+w2);
                    }
                    return true;
                }
            };


            typedef enum {EUCLIDEAN_SQR, EUCLIDEAN_EXP, EUCLIDEAN, NORM_L1, NORM_LInf, NORM_MANUAL} NormType;
        protected:
            const DataSet & m_dataset;

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
            NormType        m_norm;
            DataItem        m_weights;

            NormFunction manual_norm;
            std::shared_ptr<Averager> averager;

        public:
            DMeans(const DataSet & data, const double max_distance, const NormType norm=EUCLIDEAN_SQR);

            DMeans(const DataSet & data, const double max_distance, 
                    const DataItem & weights, const NormType norm=EUCLIDEAN_SQR);

            ~DMeans(void);

            void process(void);

            void setManualDistance(NormFunction f) {
                manual_norm = f;
            }

            void setManualAverage(std::shared_ptr<Averager> f) {
                averager = f;
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

            // Only really useful in dim < 3
            void plot(const std::string & filename, bool raw) const;

        protected:
            const DataItem & select(unsigned int i) const;
            Leave find_closest_pair(unsigned int last_elem, double dmax) const;

            static double euclidean_distance_exp(const DataItem & point1, const DataItem & point2) ;
            static double euclidean_distance_sqr(const DataItem & point1, const DataItem & point2) ;
            static double euclidean_distance(const DataItem & point1, const DataItem & point2) ;
            static double l1_distance(const DataItem & point1, const DataItem & point2) ;
            static double linf_distance(const DataItem & point1, const DataItem & point2) ;
            double distance(const DataItem & point1, const DataItem & point2) const;
            void build_label_list(std::vector<unsigned int> & labels,
                    std::vector<float> & node_pos,
                    std::vector<unsigned int> & inv_labels) const;

    };

};
#endif
