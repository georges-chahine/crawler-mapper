#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <geometry_msgs/Vector3.h>
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
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "vector"

#define PI 3.14159265
double gx=0;
double gy=0;
double gz=0;
double gxx=0;
double gyy=0;
double gzz=0;

double aa=0;
double bb=0;
double cc=0;
double dd=0;
double x0Init=0;
double x0Prev=0;
double yInit=0;
double yawInit=0;
bool init=false;
int iterator=0;
ros::Publisher pub2, pub3, pubYaw, pubSlope, pubWorld, pubWorld0, pub, pubOutliers, pubLocalOutliers;
Eigen::Matrix4d prevPose= Eigen::Matrix4d::Identity();
Eigen::Matrix4d currentPose= Eigen::Matrix4d::Identity();
double x=0;
double y=0;

std::vector<int> ransac(pcl::PointCloud<pcl::PointXYZI> p)
{
    std::vector<int> outlierIndices;
    int randMax=p.points.size();

    if (randMax > 50)
    {
        //std::cout<<"randMax is "<<randMax<<std::endl;
        int rand1, rand2, rand3;
        std::vector<std::vector<double>> log;
        //  int ii=0;
        int iterations0=500;
        int iterations=250;
        float ransacDistTh=0.2;
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

            float v1x=p2x-p1x;
            float v1y=p2y-p1y;
            float v1z=p2z-p1z;

            float v2x=p3x-p1x;
            float v2y=p3y-p1y;
            float v2z=p3z-p1z;

            float A= (v1y * v2z - v1z * v2y);
            float B= (v1z * v2x - v1x * v2z);
            float C= (v1x * v2y - v1y * v2x); //cross product

            float penalty=0;
            for (int i=0; i<iterations; i++){

                int randT=rand()%randMax;

                float x1=p.points[randT].x;
                float y1=p.points[randT].y;
                float z1=p.points[randT].z;

                penalty=(fabs(A*x1+B*y1+C*z1+1))/sqrt(A*A+B*B+C*C)+penalty;

            }
            data.push_back(A);
            data.push_back(B);
            data.push_back(C);
            data.push_back(penalty);
            log.push_back(data);

        }

        int lowest=9999999;
        float a=0; float b=0; float c=0;
        for (int i=0; i<log.size(); i++){

            if (log[i][3]<lowest){

                lowest=log[i][3];
                a=log[i][0];b=log[i][1];c=log[i][2];
            }

        }
        aa=0.9*aa+0.1*a;
        bb=0.9*bb+0.1*b;
        cc=0.9*cc+0.1*c;
        dd=0.9*dd+0.1*1;

        std::cout<<"plane equation is "<<aa<<"*x + "<<bb<<"*y + "<<cc<<"*z + "<<dd<<" = 0"<<std::endl;
        //    penalty=(fabs(A*x1+B*y1+C*z1+1))/sqrt(A*A+B*B+C*C);



        for (int i=0; i<p.points.size(); i++){

            float x1=p.points[i].x;
            float y1=p.points[i].y;
            float z1=p.points[i].z;
            float distance=(fabs(aa*x1+bb*y1+cc*z1+dd))/sqrt(aa*aa+bb*bb+cc*cc);
            if (distance>ransacDistTh) { outlierIndices.push_back(i);}

        }
    }


    return (outlierIndices);

}

void mapCb(const sensor_msgs::ImuConstPtr& imu, const nav_msgs::OdometryConstPtr& odom, const sensor_msgs::PointCloud2ConstPtr& ifm)
{
    srand (time(NULL));
    pcl::PCLPointCloud2::Ptr pc2 (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*ifm,*pc2);

    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (pc2);
    sor.setMinimumPointsNumberPerVoxel(2);
    //float leafSize=0.01f;
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*pc2);
    std::vector<int> indices;
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZI>);

    pcl::fromPCLPointCloud2(*pc2,*temp_cloud);

    pcl::removeNaNFromPointCloud(*temp_cloud, *temp_cloud, indices);

    double ax=imu->linear_acceleration.x;
    double ay=imu->linear_acceleration.y;
    double az=imu->linear_acceleration.z;
    // std::cout<<"gx is "<<gx<<" gy is "<<gy<<" gz is "<<gz<<std::endl;
    gx = 0.9 * gx + 0.1 * ax;
    gy = 0.9 * gy + 0.1 * ay;
    gz = 0.9 * gz + 0.1 * az;

    gxx = 0.95 * gxx + 0.05 * ax;
    gyy = 0.95 * gyy + 0.05 * ay;
    gzz = 0.95 * gzz + 0.05 * az;

    double x0=odom->pose.pose.position.x;

    double yaw=atan2(gy,gz);

    if (init==false)
    {
        x0Init=x0;
        x0Prev=0;
        yawInit=yaw;
        init=true;
        std::cout<<"INIT!"<<std::endl;
    }
    x0=x0-x0Init;
    double dx0=x0-x0Prev;
    x0Prev=x0;
    yaw=yaw-yawInit;

    x=x+dx0*cos(yaw);
    y=y+dx0*sin(yaw);
    //if (yaw<0){yaw=yaw+2*PI;}

    double slope=atan2(sqrt(gyy*gyy+gzz*gzz), gxx);

    double theta=slope;

    std_msgs::Float64 yawMsg;
    yawMsg.data=yaw*180/PI;

    Eigen::Matrix4d transform= Eigen::Matrix4d::Identity();

    transform(1,1)=cos(theta);
    transform(2,1)=sin(theta);
    transform(1,2)=-sin(theta);
    transform(2,2)=cos(theta);

    Eigen::Vector4d point(x,y,0.0,1.0);
    //std::cout<<"Transform is \n"<<transform<<std::endl;
    Eigen::Vector4d ImuOdom=transform*point;

    pcl::PointCloud<pcl::PointXYZ>::Ptr msg (new pcl::PointCloud<pcl::PointXYZ>);

    double xOdom=ImuOdom[0];
    double yOdom=ImuOdom[1];
    double zOdom=ImuOdom[2];

    msg->header.frame_id = "ifm2";
    msg->height = msg->width = 1;
    msg->points.push_back (pcl::PointXYZ(xOdom, yOdom, zOdom));
    pub2.publish(msg);


    pcl::PointCloud<pcl::PointXYZ>::Ptr msg2 (new pcl::PointCloud<pcl::PointXYZ>);
    msg2->header.frame_id = "ifm2";
    msg2->height = msg2->width = 1;
    msg2->points.push_back (pcl::PointXYZ(x, y, 0));
    pub3.publish(msg2);

    Eigen::Matrix4d transformRoll= Eigen::Matrix4d::Identity();
    transformRoll(1,1)=cos(theta);
    transformRoll(2,1)=sin(theta);
    transformRoll(1,2)=-sin(theta);
    transformRoll(2,2)=cos(theta);

    Eigen::Matrix4d Pose= Eigen::Matrix4d::Identity();

    Eigen::Matrix4d transformYaw= Eigen::Matrix4d::Identity();
    transformYaw(0,0)=cos(yaw);
    transformYaw(1,0)=sin(yaw);
    transformYaw(0,1)=-sin(yaw);
    transformYaw(1,1)=cos(yaw);

    Pose=transformYaw*transformRoll;

    Pose(0,3)=xOdom;
    Pose(1,3)=yOdom;
    Pose(2,3)=zOdom;


    Eigen::Matrix4d increment=prevPose.inverse()*Pose;
    if (iterator==0)
    {
        increment=Eigen::Matrix4d::Identity();
    }

    std_msgs::Float64 slopeMsg;


    pubYaw.publish(yawMsg);
    pubSlope.publish(slopeMsg);

    //Eigen::Matrix4d localTransform=transformRoll.transpose()*transformYaw;
    Eigen::Matrix4d localTransform=transformYaw;

    localTransform(0,3)=x;
    localTransform(1,3)=y;
    localTransform(2,3)=0;

    std::cout<<"local Transform is \n"<<localTransform<<std::endl;
    //  tf::Quaternion q0 = tf::createQuaternionFromRPY(-0.01137, 0.169221, 0.985511);
    tf::Quaternion q0 = tf::createQuaternionFromRPY(0, 0.34, 0);

    Eigen::Quaternion<double> rotation;
    rotation.x()=q0[0];
    rotation.y()=q0[1];
    rotation.z()=q0[2];
    rotation.w()=q0[3];
    Eigen::Matrix3d rotM=rotation.toRotationMatrix();
    Eigen::Matrix4d ifmTransform= Eigen::Matrix4d::Identity();
    ifmTransform.block(0,0,3,3) = rotM;

    ifmTransform(2,3)=0.10;

    pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr outlier_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr outlier_local_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud0 (new pcl::PointCloud<pcl::PointXYZI>);


    float boundingBox=2;
    std::cout<<"world Transform is \n"<<transformRoll*localTransform<<std::endl;

    std::vector<int> outliers = ransac(*temp_cloud);

    for (unsigned int j=0; j<temp_cloud->points.size(); j++){

        double xx=temp_cloud->points[j].x; double yy=temp_cloud->points[j].y; double zz=temp_cloud->points[j].z;
        // double wx, wy, wz;
        double intensity=temp_cloud->points[j].intensity;
        Eigen::Vector4d pcPoints(xx,yy,zz,1.0);
        Eigen::Vector4d pcPointsTransformed=localTransform*ifmTransform*pcPoints;

        Eigen::Vector4d pcPoints2(xx,yy,zz,1.0);
        Eigen::Vector4d pcPointsTransformedWorld=transformRoll*localTransform*ifmTransform*pcPoints2;


        temp_cloud->points[j].x=pcPointsTransformed[0];
        temp_cloud->points[j].y=pcPointsTransformed[1];
        temp_cloud->points[j].z=pcPointsTransformed[2];

        pcl::PointXYZI p, pLocal;
        p.x=pcPointsTransformedWorld[0];
        p.y=pcPointsTransformedWorld[1];
        p.z=pcPointsTransformedWorld[2];
        p.intensity=temp_cloud->points[j].intensity;

        pLocal.x=pcPointsTransformed[0];
        pLocal.y=pcPointsTransformed[1];
        pLocal.z=pcPointsTransformed[2];
        pLocal.intensity=temp_cloud->points[j].intensity;

        bool outlierFlag=false;


      //  if (xx>-1 && yy>-1 && zz>0.0 && xx<1 && yy<1 && zz<1) {
        if (  sqrt (xx*xx+yy*yy+zz*zz)<0.7 && zz>0) {
            //do something
        }
        else
        {


            world_cloud0->points.push_back(p);

        }


        for (int k=0; k<outliers.size(); k++){

            if (outliers[k]==j)
            {
                outlierFlag=true;
                outlier_cloud->points.push_back(p);

                outlier_local_cloud->points.push_back(pLocal);
                break;
            }
        }


        if  ((fabs(xx)<boundingBox) && (fabs(yy)<boundingBox) && (fabs(zz)<boundingBox) && (intensity>150) && (outlierFlag==false)){
            local_cloud->points.push_back(temp_cloud->points[j]);
            world_cloud->points.push_back(p);

        }
    }

    local_cloud->header.frame_id="ifm2";
    world_cloud->header.frame_id="ifm2";
    world_cloud0->header.frame_id="ifm2";
    outlier_cloud->header.frame_id="ifm2";
    outlier_local_cloud->header.frame_id="ifm2";
    pub.publish(local_cloud);
    pubWorld.publish(world_cloud);
    pubWorld0.publish(world_cloud0);
    pcl::VoxelGrid<pcl::PointXYZI> sor2;

    //sor2.setInputCloud (outlier_cloud);
    // sor2.setMinimumPointsNumberPerVoxel(2);
    //float leafSize=0.01f;
    //sor2.setLeafSize (0.01f, 0.01f, 0.01f);
    //sor2.filter (*outlier_cloud);

    pubOutliers.publish(outlier_cloud);
    pubLocalOutliers.publish(outlier_local_cloud);


    std::cout<< "yaw is "<<yaw*180/PI<<std::endl;

    std::cout<< "slope is "<<slope*180/PI<<std::endl;

    std::cout<<std::endl;
    std::cout<< "xOdom is "<<xOdom<<std::endl;
    std::cout<< "yOdom is "<<yOdom<<std::endl;
    std::cout<< "zOdom is "<<zOdom<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
    iterator++;


}

int main(int argc, char * argv[]){

    ros::init(argc, argv, "crawler_mapper");
    ros::NodeHandle n;
    std::cout<<"Syncing... "<<std::endl;

    message_filters::Subscriber<sensor_msgs::Imu> imu_sub(n, "/imu/imu", 1);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/odom_raw", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(n, "/ifm3d_filter/cloud_out", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), imu_sub, odom_sub, pc_sub);
    sync.registerCallback(boost::bind(&mapCb, _1, _2, _3));

    pub = n.advertise<pcl::PointCloud<pcl::PointXYZI>> ("local_map", 1);
    pubWorld = n.advertise<pcl::PointCloud<pcl::PointXYZI>> ("world_map", 1);
    pubWorld0 = n.advertise<pcl::PointCloud<pcl::PointXYZI>> ("world_map0", 1);
    pubOutliers = n.advertise<pcl::PointCloud<pcl::PointXYZI>> ("outliers_world_map", 1);
    pubLocalOutliers = n.advertise<pcl::PointCloud<pcl::PointXYZI>> ("outliers_local_map", 1);
    pub2 = n.advertise<pcl::PointCloud<pcl::PointXYZ>> ("imuOdom", 1);
    pub3 = n.advertise<pcl::PointCloud<pcl::PointXYZ>> ("trajectory_points", 1);
    pubYaw = n.advertise<std_msgs::Float64> ("yaw", 1);
    pubSlope = n.advertise<std_msgs::Float64> ("e2", 1);
    ros::spin();
    return 0;
}
