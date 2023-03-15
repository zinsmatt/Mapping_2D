#pragma once

#include <ros/ros.h>

#include <geometry_msgs/Vector3.h>
#include <std_msgs/Bool.h>


class PerceptionRos
{
    public:
        PerceptionRos(ros::NodeHandle& nh);

        void velocityCallback(const geometry_msgs::Vector3::ConstPtr& msg);
        void quit(const std_msgs::Bool::ConstPtr &msg);

    private:
        ros::Subscriber sub_vel_;
        ros::Subscriber  sub_quit_;
        ros::Publisher pub_pos_;

        double x_ = 0.0;
        double y_ = 0.0;
};