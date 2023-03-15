#include "perception.h"


const int QUEUE_SIZE = 10;

const int FPS = 30;

PerceptionRos::PerceptionRos(ros::NodeHandle& nh) :
    sub_vel_(nh.subscribe("/perception_vel", QUEUE_SIZE, &PerceptionRos::velocityCallback, this)),
    sub_quit_(nh.subscribe("/perception_quit", 2, &PerceptionRos::quit, this)),
    pub_pos_(nh.advertise<geometry_msgs::Vector3>("estimation_pos", QUEUE_SIZE))
{
}


void PerceptionRos::velocityCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    ROS_INFO("Velocity %f %f\n", msg->x, msg->y);

    x_ += msg->x / FPS;
    y_ += msg->y / FPS;

    geometry_msgs::Vector3 est_pos;
    est_pos.x = x_;
    est_pos.y = y_;
    est_pos.z = 0.0;

    pub_pos_.publish(est_pos);

}

void PerceptionRos::quit(const std_msgs::Bool::ConstPtr &msg) {
    ROS_INFO("Quit");
}