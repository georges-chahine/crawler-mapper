#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float64.h"
#include "nav_msgs/Odometry.h"
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <image_transport/image_transport.h>
#include "vector"
#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/IO.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "DMeansT.h"
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <tf/transform_broadcaster.h>
using namespace std;
using namespace Clustering;

#define PI 3.14159265
typedef std::pair<double,double> DI;
typedef std::vector<DI> DataSet;
typedef DataSet::const_iterator DataSetIterator;


struct DataMetric {
    double operator()(const DI & a, const DI & b) const {
        return hypot(a.first-b.first,a.second-b.second);
    }
};

struct DataAggregator {
    DI operator()(DataSetIterator ds_begin, MemberSetIterator mb_begin, MemberSetIterator mb_end) const {
        size_t count = 0;
        DI s(0,0);
        for (MemberSetIterator mb_it = mb_begin; mb_it != mb_end; mb_it++) {
            s.first += (ds_begin + *mb_it)->first;
            s.second += (ds_begin + *mb_it)->second;
            count += 1;
        }
        s.first /= count;
        s.second /= count;
        return s;
    }
};

template <class stream>
stream & operator<<(stream & out, const DI & d) {
    out << d.first << " " << d.second;
    return out;
}

class CenterSplitter {
protected:
    std::uniform_real_distribution<double> dis;
public:
    CenterSplitter() : dis(-M_PI, M_PI) {}
    template <class Generator = std::default_random_engine>
    std::pair<DI,DI> operator()(const DI & c, double scale, Generator & rge) {
        double theta = dis(rge);
        DI c1(c),c2(c);
        c1.first += scale * cos(theta);
        c1.second += scale * sin(theta);
        c2.first -= scale * cos(theta);
        c2.second -= scale * sin(theta);
        return std::pair<DI,DI>(c1,c2);
    }
};

class Mapper

{
protected:
    typedef PointMatcher<float> PM;
    //typedef PointMatcherIO<float> PMIO;
    typedef PM::TransformationParameters TP;
    typedef PM::DataPoints DP;
    //DP mapPointCloud;
    float priorDynamic;
    tf::Quaternion q0;
    tf::Vector3 t0;
    Eigen::Quaternion<double> rotation;
    Eigen::Matrix3d rotM;
    Eigen::Matrix4d ifmTransform, zeroTransform;
    //  nav_msgs::OccupancyGrid globalOutlierMap;  //obstacles occupancy grid
    float leaf_size;
    bool doIcp, skipRansac;
    double gx, gy, gz, gxx, gyy, gzz, aa, bb, cc, dd, x, y, n1, n2, n3;
    double x0Init=0;
    double x0Prev=0;
    double xOffset, yOffset;
    nav_msgs::Path path;
    image_transport::Publisher obsImagePub;
    double y0Init=0;
    double y0Prev=0;
    double yawInit=0;

    int erosion_elem = 0;
    int dilation_elem = 0;

    int erosion_size = 2;

    int dilation_size =1;

    std::string icpParamPath;
    std::string icpInputParamPath;
    std::string icpPostParamPath;
    bool init, useTf;
    bool computeProbDynamic;
    ros::NodeHandle n;
    ros::Publisher localOutlierPub, ransacPlanePub, localInlierPub, scanFilteredPub, posePub, odomPub, globalOutlierPub, pathPub;
    image_transport::ImageTransport it;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub;
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalMap, globalOutlierMap, globalMapFiltered;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;
    tf::StampedTransform ifmTf, zeroTf;
    tf::TransformListener listener1, listener2;
    tf::TransformBroadcaster broadcaster;

    void Erosion(  cv::Mat_<float>& src, unsigned int iterations )
    {


        int erosion_type = 0;
        if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
        else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
        else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }
        cv::Mat element = cv::getStructuringElement( erosion_type,
                                                     cv::Size( erosion_size + 1, erosion_size+1 ),
                                                     cv::Point( erosion_size, erosion_size ) );


        for (unsigned int i=0; i<iterations; i++){
            cv::erode( src, src, element );
        }
        //imshow( "Erosion Demo", src );
    }
    void Dilation(cv::Mat_<float>& src, unsigned int iterations )
    {
        int dilation_type = 0;
        if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
        else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
        else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }
        cv::Mat element = cv::getStructuringElement( dilation_type,
                                                     cv::Size( dilation_size + 1, dilation_size+1 ),
                                                     cv::Point( dilation_size, dilation_size ) );

        for (unsigned int i=0; i<iterations; i++){
            cv::dilate( src, src, element );
        }
        //  imshow( "Dilation Demo", src );
    }



    pcl::PointCloud<pcl::PointXYZI> boundingBox(pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud, double insideTh, double outsideTh){

        pcl::PointCloud<pcl::PointXYZI> p;

        for (unsigned int j=0; j<temp_cloud->points.size(); j++){

            double xx=temp_cloud->points[j].x;
            double yy=temp_cloud->points[j].y;
            double zz=temp_cloud->points[j].z;

            if (  (sqrt (xx*xx+yy*yy+zz*zz)<insideTh && zz>0) || sqrt (xx*xx+yy*yy+zz*zz)>outsideTh || zz>0.5 ) {  //hemispherical box
                //do nothing for now
            }
            else
            {
                p.points.push_back(temp_cloud->points[j]);
            }
        }
        return p;
    }

    Eigen::Matrix4d getPose(const sensor_msgs::ImuConstPtr& imu, const nav_msgs::OdometryConstPtr& odom){

        double ax=imu->linear_acceleration.x;
        double ay=imu->linear_acceleration.y;
        double az=imu->linear_acceleration.z;



        double tx=odom->pose.pose.position.x;
        double ty=odom->pose.pose.position.y;
        double tz=odom->pose.pose.position.z;

        double qx=odom->pose.pose.orientation.x;
        double qy=odom->pose.pose.orientation.y;
        double qz=odom->pose.pose.orientation.z;
        double qw=odom->pose.pose.orientation.w;

        Eigen::Quaterniond q(qw,qx,qy,qz);
        //Eigen::Matrix3d M(q);

        auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);



        gx = 0.6 * gx + 0.4* ax;
        gy = 0.6 * gy + 0.4* ay;
        gz = 0.6 * gz + 0.4* az;
        gxx = 0.95 * gxx + 0.05 * ax;
        gyy = 0.95 * gyy + 0.05 * ay;
        gzz = 0.95 * gzz + 0.05 * az;

        double x0=odom->pose.pose.position.x;


        double y0=odom->pose.pose.position.y;

        x0=sqrt(x0*x0+y0*y0);

        double yaw=euler(2);
        //std::cout<<"euler 0 "<<euler(0)<<"euler 1 "<<euler(1)<<"euler 2 "<<euler(2)<<std::endl;
        //double yaw=-atan2(gy,gz);

        if (init==false)
        {
            x0Init=x0;   //start at x=1
            x0Prev=0;

            y0Init=x0;   //start at y=1
            y0Prev=0;


            yawInit=yaw;
            init=true;
            std::cout<<"INIT!"<<std::endl;
        }

        x0=x0-x0Init;
        double dx0=x0-x0Prev;
        x0Prev=x0;

        y0=y0-y0Init;
        double dy0=y0-y0Prev;
        y0Prev=y0;

        yaw=yaw-yawInit;

        x=x+dx0*cos(yaw);
        y=y+dx0*sin(yaw);

        //if (yaw<0){yaw=yaw+2*PI;}

        double slope=atan2(sqrt(gyy*gyy+gzz*gzz), gxx);

        std_msgs::Float64 yawMsg;
        yawMsg.data=yaw*180/PI;

        Eigen::Vector4d point(x,y,0.0,1.0);
        //std::cout<<"Transform is \n"<<transform<<std::endl;

        Eigen::Matrix4d transformRoll= Eigen::Matrix4d::Identity();
        transformRoll(1,1)=cos(slope);
        transformRoll(2,1)=sin(slope);
        transformRoll(1,2)=-sin(slope);
        transformRoll(2,2)=cos(slope);

        Eigen::Matrix4d transformYaw= Eigen::Matrix4d::Identity();
        transformYaw(0,0)=cos(yaw);
        transformYaw(1,0)=sin(yaw);
        transformYaw(0,1)=-sin(yaw);
        transformYaw(1,1)=cos(yaw);

        Eigen::Matrix4d localTransform=transformYaw;

        localTransform(0,3)=x;
        localTransform(1,3)=y;
        localTransform(2,3)=0;




        if (useTf){


            // tf::StampedTransform temp;

            // listener1.waitForTransform("odom_raw", "base_link", odom->header.stamp ,ros::Duration(1.0) );

            // listener1.lookupTransform("odom_raw", "base_link", ros::Time(0), temp);

            //tf::Quaternion qq=temp.getRotation();
            //tf::Vector3 t=temp.getOrigin();

            localTransform(0,3)=tx;
            localTransform(1,3)=ty;
            localTransform(2,3)=tz;

            Eigen::Quaterniond qqq(qw,qx,qy,qz);
            Eigen::Matrix3d mmm(qqq);

            localTransform.block(0,0,3,3)=mmm;


        }

        localTransform=zeroTransform*localTransform;










        return localTransform;




    }

    void ransac(pcl::PointCloud<pcl::PointXYZI> p, float cx, float cy,float ransacDistTh, std::vector<double>& planeParameters,  std::vector<int>& outlierIndices, std::vector<int>& inlierIndices, bool skip)
    {

        double maxPenalty=1111;
        int randMax=p.points.size();

        if (randMax > 1000)
        {
            //std::cout<<"randMax is "<<randMax<<std::endl;
            int rand1, rand2, rand3;
            std::vector<std::vector<double>> log;
            //  int ii=0;
            int iterations0=500;
            int iterations=500;
            if (skip){

                iterations=iterations0=1;
            }
            for (int j=0; j<iterations0 ; j++){

                std::vector<double> data;

                bool randCheck=false;
                while (randCheck ==false)
                {
                    rand1=rand()%randMax;
                    rand2=rand()%randMax;
                    rand3=rand()%randMax;
                    if ((rand1 != rand2) && (rand2 != rand3) && (rand1 != rand3)) {randCheck=true;}
                }
                float p1x=p.points[rand1].x;
                float p2x=p.points[rand2].x;
                float p3x=p.points[rand3].x;


                float p1y=p.points[rand1].y;
                float p2y=p.points[rand2].y;
                float p3y=p.points[rand3].y;

                float p1z=p.points[rand1].z;
                float p2z=p.points[rand2].z;
                float p3z=p.points[rand3].z;


                //std::cout<<p1x<<" "<<p1y<<" "<<p1z<<","<<p2x<<" "<<p2y<<" "<<p2z<<","<<p3x<<" "<<p3y<<" "<<p3z<<std::endl;
                float v1x=p2x-p1x;
                float v1y=p2y-p1y;
                float v1z=p2z-p1z;

                float v2x=p3x-p1x;
                float v2y=p3y-p1y;
                float v2z=p3z-p1z;

                float A= (v1y * v2z - v1z * v2y);
                float B= (v1z * v2x - v1x * v2z);
                float C= (v1x * v2y - v1y * v2x); //cross product

                A=A/sqrt(A*A+B*B+C*C);
                B=B/sqrt(A*A+B*B+C*C);
                C=C/sqrt(A*A+B*B+C*C);




                float D=-A*p1x-B*p1y-C*p1z;
                //A=A/D;
                //%B=B/D;
                //C=C/D;
                //D=1;

                double penalty=0;
                for (int i=0; i<iterations; i++){

                    int randT=rand()%randMax;
                    // std::cout<<"randT is "<<randT<<" p.size() is"<<p.points.size()<<std::endl;

                    float x1=p.points[randT].x;
                    float y1=p.points[randT].y;
                    float z1=p.points[randT].z;

                    double distance=(std::abs(A*x1+B*y1+C*z1+D))/sqrt(A*A+B*B+C*C);
                    //std::cout<<distance<<std::endl;
                    if (distance>ransacDistTh){

                        penalty=penalty+maxPenalty;


                    }
                    else{

                        penalty=penalty+distance;
                        //std::cout<<std::setprecision(16)<<"penalty is "<<penalty<<std::endl;

                    }

                }



                A=0; B=0; C=1;  //override ransac
                data.push_back(A);
                data.push_back(B);
                data.push_back(C);
                data.push_back(penalty);


                data.push_back(D);
                log.push_back(data);

            }
            /*for (int i=0; i<log.size(); i++){


                std::cout<<std::setprecision(16)<<"log: "<<log[i][0]<<","<<log[i][1]<<","<<log[i][2]<<" penalty:"<<log[i][3]<<std::endl;

                }
            */

            double lowest=9999999999;
            float a=0; float b=0; float c=0; float d=0;
            for (int i=0; i<log.size(); i++){

                if (log[i][3]<lowest){

                    lowest=log[i][3];
                    a=log[i][0];b=log[i][1];c=log[i][2]; d=log[i][4];
                }

            }




            if (aa==0 && bb==0 && cc==0 && dd==0){
                aa=a;
                bb=b;
                cc=c;
                dd=-aa*cx-bb*cy-cc*0;
            }

            else

            {
                aa=a;
                bb=b;
                cc=c;
                //  dd=d;
                //aa=0.99*aa+0.01*a;
                // bb=0.99*bb+0.01*b;
                // cc=0.99*cc+0.01*c;
                //dd=-aa*lastpx-bb*lastpy-cc*lastpz;
                dd=-aa*cx-bb*cy-cc*0;


                //  dd=0.99*dd+0.01*d;



            }


            // std::cout<<"plane equation is "<<aa<<"*x + "<<bb<<"*y + "<<cc<<"*z + "<<dd<<" = 0"<<std::endl;

            int counter=0;
            for (int i=0; i<p.points.size(); i++){

                float x1=p.points[i].x;
                float y1=p.points[i].y;
                float z1=p.points[i].z;
                float distance=(std::abs(aa*x1+bb*y1+cc*z1+dd))/sqrt(aa*aa+bb*bb+cc*cc);
                if (distance>ransacDistTh  &&  z1 > ((-dd-aa*x1-bb*y1)/cc)  ) {
                    outlierIndices.push_back(i);
                    counter++;


                }
                else
                {

                    inlierIndices.push_back(i);

                }

            }

            std::cout<<"points out of plane "<<counter<<" out of "<<p.points.size()<<" ("<<(100*counter)/p.points.size()<<"%)"<<std::endl;
        }



        planeParameters[0]=aa; planeParameters[1]=bb;planeParameters[2]=cc;planeParameters[3]=dd;

        std::cout<<"plane parameters \n"<<planeParameters[0]<<" "<<planeParameters[1]<<" "<<planeParameters[2]<<" "<<planeParameters[3]<<std::endl;



        pcl::PointCloud<pcl::PointXYZ> pPlane;
        pcl::PointXYZ pTemp;
        for (float i=cx-1; i<cx+1; i=i+0.1){
            for (float j=cy-1; j<cy+1; j=j+0.1)
            {
                pTemp.x=i;
                pTemp.y=j;
                pTemp.z=(-dd-aa*pTemp.x-bb*pTemp.y)/cc;
                pPlane.points.push_back(pTemp);
            }

        }
        pPlane.height=1;
        pPlane.width=pPlane.points.size();
        pPlane.header.frame_id = "map";

        ransacPlanePub.publish(pPlane);



    }

    void getLocalMap(Eigen::Matrix4d pose,pcl::PointCloud<pcl::PointXYZI>::Ptr globalMap, pcl::PointCloud<pcl::PointXYZI>::Ptr& localMap,float radius){


        for (int i=0; i<globalMap->points.size(); i++){
            pcl::PointXYZI p=globalMap->points[i];

            double x=pose(0,3);
            double y=pose(1,3);
            double z=pose(2,3);


            double distance=sqrt(pow(p.x-x,2)+pow(p.y-y,2)+pow(p.z-z,2));

            if (distance<radius){
                p.x=p.x;  //-x;
                p.y=p.y;  //-y;
                Eigen::Vector4d ptTransformed(p.x,p.y,p.z,1.0);
                ptTransformed=pose.inverse()*ptTransformed;
                p.x=ptTransformed[0];
                p.y=ptTransformed[1];
                p.z=ptTransformed[2];
                localMap->points.push_back(p);
                // localStamps.push_back(stamps[i])

            }
        }




    }


    void collapsePoint(pcl::PointXYZI& p, std::vector<double> planeParameters){

        double x= p.x;
        double y= p.y;
        double z= p.z;
        double planeZ=(-planeParameters[0]*x-planeParameters[1]*y-planeParameters[3])/planeParameters[2];

        if (planeZ>z){
             //   std::cout<<"plane z is "<<planeZ<<std::endl;
            p.z=planeZ; //collapse points below the plane onto the plane
        }
    }

    void collapsePoints(pcl::PointCloud<pcl::PointXYZI>& cloudIn, std::vector<double> planeParameters){

        for (int i=0; i<cloudIn.points.size(); i++){   //collapse points below the ransac plane

            double x= cloudIn.points[i].x;
            double y= cloudIn.points[i].y;
            double z= cloudIn.points[i].z;

            double planeZ=(-planeParameters[0]*x-planeParameters[1]*y-planeParameters[3])/planeParameters[2];

            if (planeZ>z){
                //    std::cout<<"plane z is "<<planeZ<<std::endl;
                cloudIn.points[i].z=planeZ; //collapse points below the plane onto the plane

            }



        }

    }

    void icpFn(DP& newCloud, DP& mapPointCloud, Eigen::Matrix4d& correction, Eigen::Matrix4d& prior, std::string icpParamPath, std::string icpInputParamPath, std::string icpPostParamPath)
    {

        correction=Eigen::Matrix4d::Identity();
        // Rigid transformation
        std::shared_ptr<PM::Transformation> rigidTrans;
        rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

        // Main algorithm definition
        PM::ICP icp;
        PM::DataPointsFilters inputFilters;
        PM::DataPointsFilters mapPostFilters;

        if(!icpParamPath.empty())
        {
            std::ifstream ifs(icpParamPath.c_str());
            icp.loadFromYaml(ifs);
            std::cout<<"loaded icp yaml!"<<std::endl;
            ifs.close();
        }
        else
        {
            icp.setDefault();
        }

        if(!icpInputParamPath.empty())
        {
            std::ifstream ifs(icpInputParamPath.c_str());
            inputFilters = PM::DataPointsFilters(ifs);
            std::cout<<"loaded input filter yaml!"<<std::endl;
            ifs.close();
        }

        if(!icpPostParamPath.empty())
        {
            std::ifstream ifs(icpPostParamPath.c_str());
            mapPostFilters = PM::DataPointsFilters(ifs);
            std::cout<<"loaded post filter yaml!"<<std::endl;
            ifs.close();
        }

        inputFilters.apply(newCloud);

        std::cout<<"map pc has "<<mapPointCloud.getNbPoints()<<"points"<<std::endl;
        mapPostFilters.apply(mapPointCloud);
        if (newCloud.getNbPoints()>50){


            try
            {
                // We use the last transformation as a prior
                // this assumes that the point clouds were recorded in
                // sequence.
                const TP prior0 = prior.cast<float>();
                // const TP prior = T_to_map_from_new*initialEstimate.matrix().cast<float>();
                //const TP prior= TP::Identity(4,4);

                TP T_to_map_from_new = icp(newCloud, mapPointCloud, prior0);

                T_to_map_from_new = rigidTrans->correctParameters(T_to_map_from_new);
                newCloud = rigidTrans->compute(newCloud, T_to_map_from_new);
                // mapPointCloud.concatenate(newCloud);

                correction=T_to_map_from_new.cast <double> ();

            }
            catch (PM::ConvergenceError& error)
            {
                std::cout << "ERROR PM::ICP failed to converge: " << std::endl;
                std::cout << "   " << error.what() << std::endl;

            }

        }
    }

    void transform_cloud(pcl::PointCloud<pcl::PointXYZI>& cloudIn,Eigen::Matrix4d pose){

        for (unsigned int j=0; j<cloudIn.points.size(); j++){

            double xx=cloudIn.points[j].x; double yy=cloudIn.points[j].y; double zz=cloudIn.points[j].z;
            // double wx, wy, wz;
            double intensity=cloudIn.points[j].intensity;
            Eigen::Vector4d pcPoints(xx,yy,zz,1.0);
            Eigen::Vector4d pcPointsTransformed=pose*pcPoints;
            pcl::PointXYZI p;
            p.x=pcPointsTransformed[0];
            p.y=pcPointsTransformed[1];
            p.z=pcPointsTransformed[2];
            p.intensity=cloudIn.points[j].intensity;

            cloudIn.points[j]=p;
        }

    }


    cv::Mat_<float> publishObstacleLayer(pcl::PointCloud<pcl::PointXYZI>& cloudIn, double cx, double cy, ros::Time stamp, float radius, float resolution){



        pcl::PointCloud<pcl::PointXYZI> cloudOut;

        int w= int((2*radius)/resolution);

        cv::Mat_<float> depthFrame(w,w,1.0);

        for (int i=0; i<cloudIn.points.size(); i++){

            int v=int(  (cloudIn.points[i].x+ radius)  *(1/resolution));
            // u=w-u; //computer vision convention, x points to the right, y points down.
            int u=int(  (cloudIn.points[i].y+ radius)  *(1/resolution));

            if (u>w || v>w ||u<0 ||v<0){
                std::cout<<"u "<<u<<" v "<<v <<std::endl;
            }
            depthFrame(u,v)=0.0; //cloudIn.points[i].z;
            //            std::cout<<depthFrame(u,v)<<std::endl;

        }


        double min;
        double max;
        cv::minMaxIdx(depthFrame, &min, &max);

        Dilation (depthFrame,1);
        Erosion  (depthFrame,1);

        for (int i=0; i<cloudIn.points.size(); i++){

            int v=int(  (cloudIn.points[i].x+ radius)  *(1/resolution));
            // u=w-u; //computer vision convention, x points to the right, y points down.
            int u=int(  (cloudIn.points[i].y+ radius)  *(1/resolution));

            if (depthFrame(u,v)==0.0){


                cloudOut.points.push_back(cloudIn.points[i] );
            }
        }
        std::cout<< "min is "<<min<<" max is "<<max<<" w is "<<w<<" size is "<<cloudIn.points.size()<<std::endl;
        cloudIn=cloudOut;


        //Dilation(depthFrame);


        //std::cout<<depthFrame(u,v)<<std::endl;



        sensor_msgs::ImagePtr msgdepth;
        std::cout<<"0"<<std::endl;
        msgdepth= cv_bridge::CvImage(std_msgs::Header(), "32FC1", depthFrame).toImageMsg();
        std::cout<<"1"<<std::endl;
        msgdepth->header.stamp=stamp;
        msgdepth->header.frame_id="base_link";
        if(!depthFrame.empty()){
            obsImagePub.publish(msgdepth);
        }
        std::cout<<"2"<<std::endl;
        return depthFrame;

    }

    void updateOccupancy(pcl::PointCloud<pcl::PointXYZI>& cloudIn,std_msgs::Header header , double px, double py, double th, float resolution  ){



        /*globalOutlierMap.header=header;
        globalOutlierMap.header.frame_id="odom_raw";
        globalOutlierMap.info.map_load_time=header.stamp;
        globalOutlierMap.info.resolution=resolution;*/
        for(int i=0;i<cloudIn.points.size();i++){

            pcl::PointXYZI p;
            p.x=cloudIn.points[i].x+px;
            p.y=cloudIn.points[i].y+py;
            p.z=cloudIn.points[i].z;
            p.intensity=cloudIn.points[i].intensity;

            bool found=false;
            for (int j=0; j<globalOutlierMap->points.size(); j++){

                if (globalOutlierMap->points[i].x==p.x && globalOutlierMap->points[i].y==p.y){

                    found =true;
                    break;

                }

            }
            if (!found){

                globalOutlierMap->points.push_back(  p  );

            }


        }







    }

    void dMeansFn(pcl::PointCloud<pcl::PointXYZI>& cloudIn){


        typedef DMeansT<DI,DataSetIterator,DataMetric,DataAggregator> DMeans2D;

        DataSet num_test;

        for (int i=0; i<cloudIn.points.size();i++){

            num_test.push_back(DI( cloudIn.points[i].x, cloudIn.points[i].y));

        }


        cout << "Number of instances: " << num_test.size() << endl;

        DMeans2D::CenterSet initial_centers;
        initial_centers.push_back(DI(0, 0));
        // initial_centers.push_back(DI(1, -4));

        cout << "running d-means " << endl;

        //minimum delta between points? not sure
        DataMetric metric;
        DataAggregator aggregator;
        DMeans2D x(num_test.begin(),num_test.end(), metric, aggregator, 10);

        x.process();

        const ClusterSet & clusters = x.clusters() ;
        unsigned int n_clusters = clusters.size();

        std::ofstream ocenters ("center", std::ofstream::out);
        std::ofstream odata ("data", std::ofstream::out);
        cout << "D-MEANS found " << n_clusters << " clusters." << endl;
        for(unsigned int i=0; i<n_clusters; i++){
            const Membership & c = clusters[i];
            ocenters << x.centers()[i] << " " << i << std::endl;
            cout << "CLUSTER "<<i<<" centered on " << x.centers()[i] << " has ";
            for(size_t j=0; j<c.size(); j++){
                if (j>0) {
                    //    cout << ", " ;
                }
                //  cout << "(" << num_test[c[j]] << ") ";
                odata << num_test[c[j]] << " " << i << std::endl;
            }
            cout << endl;
        }
        ocenters.close();
        odata.close();

    }

    void initFn(std::string frame_id){


        try
        {
            listener1.waitForTransform("base_link", frame_id, ros::Time(0), ros::Duration(1.0) );

            listener1.lookupTransform("base_link", frame_id, ros::Time(0), ifmTf);

            tf::Quaternion q=ifmTf.getRotation();
            tf::Vector3 t=ifmTf.getOrigin();
            ifmTransform(0,3)=t.x();
            ifmTransform(1,3)=t.y();
            ifmTransform(2,3)=t.z()-0.05;

            Eigen::Quaterniond qq(q.w(),q.x(),q.y(),q.z());
            Eigen::Matrix3d mm(qq);

            ifmTransform.block(0,0,3,3)=mm;





            std::cout<<"tf retrieved, ifm to base is \n" <<ifmTransform<<std::endl;



        }

        catch (tf::TransformException& ex)
        {
            ROS_ERROR("Received an exception trying to transform: %s", ex.what());
        }




        try
        {


            listener1.waitForTransform("odom_raw", "base_link", ros::Time(0), ros::Duration(1.0) );

            listener1.lookupTransform("odom_raw", "base_link", ros::Time(0), zeroTf);

            tf::Quaternion q=zeroTf.getRotation();
            tf::Vector3 t=zeroTf.getOrigin();
            zeroTransform(0,3)=t.x();
            zeroTransform(1,3)=t.y();
            zeroTransform(2,3)=t.z();

            Eigen::Quaterniond qqq(q.w(),q.x(),q.y(),q.z());
            Eigen::Matrix3d mmm(qqq);

            zeroTransform.block(0,0,3,3)=mmm;
            zeroTransform=zeroTransform.inverse().eval();
            zeroTransform(0,3)=zeroTransform(0,3)+xOffset;
            zeroTransform(1,3)=zeroTransform(1,3)+yOffset;

            std::cout<<"init tf retrieved, 0 to base is \n" <<zeroTransform<<std::endl;


        }



        catch (tf::TransformException& ex)
        {
            ROS_ERROR("Received an exception trying to transform: %s", ex.what());
        }

    }

    void updateTf(ros::Time time){
        Eigen::Matrix4d tempMat=zeroTransform.inverse();

        Eigen::Matrix3d tempM=tempMat.block(0,0,3,3);
        Eigen::Quaterniond tempQ(tempM);
        tf::Quaternion qCam1(tempQ.x(),tempQ.y(),tempQ.z(),tempQ.w());
        broadcaster.sendTransform(
                    tf::StampedTransform(
                        tf::Transform(qCam1, tf::Vector3(tempMat(0,3),tempMat(1,3),tempMat(2,3) )),
                        time,"odom_raw", "map"));
    }
    void mapCb(const sensor_msgs::ImuConstPtr& imu, const nav_msgs::OdometryConstPtr& odom, const sensor_msgs::PointCloud2ConstPtr& ifm)
    {
        if (init==false && useTf)
        {
            initFn(ifm->header.frame_id);

        }
        updateTf(ifm->header.stamp);

        double insideTh=0.1;  //bounding box sphere, delete inside
        double outsideTh=1;   //bounding box sphere, delete outside
        float radius=2;  //local map radius
        float ransac_threshold=0.01;

        //read, voxelize and filter NAN
        pcl::PCLPointCloud2::Ptr pc2 (new pcl::PCLPointCloud2 ());
        pcl_conversions::toPCL(*ifm,*pc2);
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        sor.setInputCloud (pc2);
        sor.setMinimumPointsNumberPerVoxel(2);
        sor.setLeafSize (leaf_size,leaf_size,leaf_size);
        sor.filter (*pc2);

        std::vector<int> indices;
        pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromPCLPointCloud2(*pc2,*temp_cloud);
        pcl::removeNaNFromPointCloud(*temp_cloud, *temp_cloud, indices);

        Eigen::Matrix4d pose=getPose(imu, odom);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn (new pcl::PointCloud<pcl::PointXYZI>);

        *cloudIn= boundingBox(temp_cloud, insideTh, outsideTh);

        pcl::PointCloud<pcl::PointXYZI>::Ptr localMap (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr localOutlierMap (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr localInlierMap (new pcl::PointCloud<pcl::PointXYZI>);

        std::vector<double> planeDefNormal{0,0,1};

        double d=(-planeDefNormal[0]*pose(0,3)-planeDefNormal[1]*pose(1,3)-planeDefNormal[2]*(pose(2,3)));


        std::vector<double> planeDefault{planeDefNormal[0],planeDefNormal[1],planeDefNormal[2],d};

        transform_cloud(*cloudIn, pose*ifmTransform);
        //transform_cloud(*cloudIn, ifmTransform);
        collapsePoints(*cloudIn, planeDefault);


        //  globalMap->points.push_back(p);

        if (doIcp && cloudIn->points.size()>500){
            using namespace PointMatcherSupport;

            pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;

            ne.setInputCloud (cloudIn);
            // Create an empty kdtree representation, and pass it to the normal estimation object.
            // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
            pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
            ne.setSearchMethod (tree);
            // Output datasets
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
            // Use all neighbors in a sphere of radius 3cm
            ne.setRadiusSearch (0.03);
            // Compute the features
            ne.compute (*cloud_normals);

            DP data, mapPointCloud;

            Eigen::MatrixXf dataNormals(3,cloud_normals->getMatrixXfMap(3,8,0).row(0).size());
            dataNormals.row(0)=cloud_normals->getMatrixXfMap(3,8,0).row(0);
            dataNormals.row(1)=cloud_normals->getMatrixXfMap(3,8,0).row(1);
            dataNormals.row(2)=cloud_normals->getMatrixXfMap(3,8,0).row(2);

            Eigen::MatrixXf datax(1,cloudIn->getMatrixXfMap(3,8,0).row(0).size());
            Eigen::MatrixXf datay(1,cloudIn->getMatrixXfMap(3,8,0).row(1).size());
            Eigen::MatrixXf dataz(1,cloudIn->getMatrixXfMap(3,8,0).row(2).size());

            datax=cloudIn->getMatrixXfMap(3,8,0).row(0);
            datay=cloudIn->getMatrixXfMap(3,8,0).row(1);
            dataz=cloudIn->getMatrixXfMap(3,8,0).row(2);

            data.addFeature("x", datax);
            data.addFeature("y", datay);
            data.addFeature("z", dataz);
            data.addDescriptor("normals", dataNormals);

            ne.setInputCloud (globalMap);
            // Create an empty kdtree representation, and pass it to the normal estimation object.
            // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
            pcl::search::KdTree<pcl::PointXYZI>::Ptr tree1 (new pcl::search::KdTree<pcl::PointXYZI> ());
            ne.setSearchMethod (tree1);
            // Output datasets
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1 (new pcl::PointCloud<pcl::Normal>);
            // Use all neighbors in a sphere of radius 3cm
            ne.setRadiusSearch (0.03);
            // Compute the features
            ne.compute (*cloud_normals1);

            Eigen::MatrixXf dataNormals1(3,cloud_normals1->getMatrixXfMap(3,8,0).row(0).size());
            dataNormals1.row(0)=cloud_normals1->getMatrixXfMap(3,8,0).row(0);
            dataNormals1.row(1)=cloud_normals1->getMatrixXfMap(3,8,0).row(1);
            dataNormals1.row(2)=cloud_normals1->getMatrixXfMap(3,8,0).row(2);

            Eigen::MatrixXf datax1(1,globalMap->getMatrixXfMap(3,8,0).row(0).size());
            Eigen::MatrixXf datay1(1,globalMap->getMatrixXfMap(3,8,0).row(1).size());
            Eigen::MatrixXf dataz1(1,globalMap->getMatrixXfMap(3,8,0).row(2).size());

            datax1=globalMap->getMatrixXfMap(3,8,0).row(0);
            datay1=globalMap->getMatrixXfMap(3,8,0).row(1);
            dataz1=globalMap->getMatrixXfMap(3,8,0).row(2);

            mapPointCloud.addFeature("x", datax1);
            mapPointCloud.addFeature("y", datay1);
            mapPointCloud.addFeature("z", dataz1);
            mapPointCloud.addDescriptor("normals", dataNormals1);


            if(computeProbDynamic)

            {
                data.addDescriptor("probabilityDynamic", PM::Matrix::Constant(1, data.features.cols(), priorDynamic));

            }
            if(mapPointCloud.getNbPoints()  == 0)
            {
                mapPointCloud = data;

            }



            Eigen::Matrix4d correction;
            Eigen::Matrix4d prior=Eigen::Matrix4d::Identity();
            icpFn(data, mapPointCloud, correction, prior, icpParamPath, icpInputParamPath, icpPostParamPath);
            std::cout<<"correction is \n"<<correction<<std::endl;

            if (!correction.isIdentity()){
                transform_cloud(*cloudIn, correction);
                pose=pose*correction;
            }
            /*else{
                transform_cloud(*cloudIn, pose);


            }*/


        }


        *globalMap += *cloudIn;

        globalMap->header.frame_id = "map";

        //globalMap->header.stamp=cloudIn->header.stamp;


        globalMap->height = 1;
        globalMap->width = globalMap->points.size();

        getLocalMap(pose,globalMap, localMap,radius);



        /*pcl::StatisticalOutlierRemoval<pcl::PointXYZI> statFilter;
        statFilter.setInputCloud (localMap);
        statFilter.setMeanK (10);
        statFilter.setStddevMulThresh (4.5);
        statFilter.filter (*localMap);*/


        std::vector<double> planeParameters{0,0,0,0};
        std::vector<int> outliers, inliers;

        ransac(*localMap, pose(0,3), pose(1,3), ransac_threshold, planeParameters, outliers, inliers, skipRansac);

        double p1=planeParameters[0];
        double p2=planeParameters[1];
        double p3=planeParameters[2];
        double p4=planeParameters[3];
        std::cout<<"plane equation is "<<p1<<"*x + "<<p2<<"*y + "<<p3<<"*z + "<<p4<<" = 0"<<std::endl;

        for (int k=0; k<outliers.size(); k++){

            {
                localOutlierMap->points.push_back(localMap->points[outliers[k]]);
            }
        }

        for (int k=0; k<inliers.size(); k++){

            {
                localInlierMap->points.push_back(localMap->points[inliers[k]]);


            }
        }

        // *globalOutlierMap += *localOutlierMap;
        //localOutlierMap->header=ifm->header;

        cv::Mat_<float> localImg = publishObstacleLayer(*localOutlierMap, pose(0,3), pose(1,3), ifm->header.stamp, radius, leaf_size);


        updateOccupancy(*localOutlierMap, ifm->header, pose(0,3),pose(1,3),outsideTh, leaf_size);



        //dMeansFn(*globalOutlierMap);

        globalOutlierMap->header.frame_id = "map";
        globalOutlierMap->height = 1;
        globalOutlierMap->width = globalOutlierMap->points.size();

        globalOutlierPub.publish(globalOutlierMap);


        //updateMap(localImg, pose)
        localOutlierMap->header.frame_id = "base_link";
        pcl_conversions::toPCL(ifm->header.stamp, localOutlierMap->header.stamp);
        localOutlierMap->height = 1;
        localOutlierMap->width = localOutlierMap->points.size();
        localOutlierPub.publish(localOutlierMap);

        localInlierMap->header.frame_id = "base_link";
        pcl_conversions::toPCL(ifm->header.stamp, localInlierMap->header.stamp);
        localInlierMap->height = 1;
        localInlierMap->width = localInlierMap->points.size();
        localInlierPub.publish(localInlierMap);

        /*globalMap->header.frame_id = "map";
        globalMap->height = 1;
        globalMap->width = globalMap->points.size();
        pcl::VoxelGrid<pcl::PointXYZI> sor2;
        sor2.setInputCloud (globalMap);
        sor2.setMinimumPointsNumberPerVoxel(1);
        sor2.setLeafSize (leaf_size,leaf_size,leaf_size);
        sor2.filter (*globalMap);
*/
        cloudIn->header.frame_id = "map";
        cloudIn->height = 1;
        cloudIn->width = cloudIn->points.size();
        scanFilteredPub.publish(cloudIn);

        geometry_msgs::PoseStamped poseStamped;
        poseStamped.header=ifm->header;
        poseStamped.header.frame_id = "map";
        poseStamped.pose.position.x=pose(0,3);
        poseStamped.pose.position.y=pose(1,3);
        Eigen::Matrix3d dcm;
        dcm=pose.block(0,0,3,3);
        Eigen::Quaterniond q(dcm);
        poseStamped.pose.orientation.x=q.x();
        poseStamped.pose.orientation.y=q.y();
        poseStamped.pose.orientation.z=q.z();
        poseStamped.pose.orientation.w=q.w();
        posePub.publish(poseStamped);


        path.header=poseStamped.header;
        path.header.frame_id="map";
        path.poses.push_back(poseStamped);
        pathPub.publish(path);

        nav_msgs::Odometry odom_new;

        odom_new.header=ifm->header;
        odom_new.header.frame_id = "map";
        odom_new.pose.pose.position.x=pose(0,3);
        odom_new.pose.pose.position.y=pose(1,3);
        odom_new.pose.pose.orientation.x=q.x();
        odom_new.pose.pose.orientation.y=q.y();
        odom_new.pose.pose.orientation.z=q.z();
        odom_new.pose.pose.orientation.w=q.w();

        odomPub.publish(odom_new);


    }


public:
    Mapper() : n("~"), it(n) {



        xOffset=1.5;
        yOffset=1;
        icpParamPath="/home/gchahine/catkin_ws/src/crawler_mapper/icpConfig/icp_param.yaml";
        icpInputParamPath="/home/gchahine/catkin_ws/src/crawler_mapper/icpConfig/input_filters.yaml";
        icpPostParamPath="/home/gchahine/catkin_ws/src/crawler_mapper/icpConfig/mapPost_filters.yaml";
        computeProbDynamic=true;
        doIcp=false;
        useTf=true;
        skipRansac=true;
        gx=gy=gz=gxx=gyy=gzz=aa=bb=cc=dd=x=y=n1=n2=n3=0;
        leaf_size=0.025;


        q0 = tf::createQuaternionFromRPY(0, 0.33, 0);
        rotation.x()=q0[0];
        rotation.y()=q0[1];
        rotation.z()=q0[2];
        rotation.w()=q0[3];
        rotM=rotation.toRotationMatrix();
        ifmTransform=zeroTransform=Eigen::Matrix4d::Identity();
        ifmTransform.block(0,0,3,3) = rotM;

        ifmTransform(2,3)=0.14;

        init=false;
        globalMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        globalMapFiltered = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        globalOutlierMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);


        srand (time(NULL));
        ros::Duration(0.5).sleep();
        std::string transport = "raw";
        n.param("transport",transport,transport);

        imu_sub.subscribe(n, "/imu/imu", 1);
        odom_sub.subscribe(n, "/odom_raw", 1);
        pc_sub.subscribe(n, "/ifm3d_filter/cloud_out", 1);

        localOutlierPub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("local_outlier_map", 1);
        localInlierPub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("local_inlier_map", 1);

        pathPub = n.advertise<nav_msgs::Path > ("path_map", 1);

        globalOutlierPub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("global_outlier_pc", 1);

        scanFilteredPub = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("scan_filtered", 1);
        posePub = n.advertise<geometry_msgs::PoseStamped> ("pose_stamped", 1);
        odomPub = n.advertise<nav_msgs::Odometry> ("mapper_odom", 1);

        obsImagePub = it.advertise("obs_img", 1);


        ransacPlanePub = n.advertise<pcl::PointCloud<pcl::PointXYZ> > ("ransac_plane", 1);
        sync.reset(new Sync(MySyncPolicy(10), imu_sub, odom_sub, pc_sub));

        sync->registerCallback(boost::bind(&Mapper::mapCb, this, _1, _2, _3));


    }

};


int main(int argc, char * argv[]){

    ros::init(argc, argv, "bw2_mapper");

    Mapper var;
    ros::spin();

}
