#include <iostream>

#include <ros/ros.h>

#include "perception.h"

int main(int argc, char **argv) {

    ros::init(argc, argv, "perception_node");
    ros::NodeHandle nh;

    std::cout << "LOSC\n";

    PerceptionRos perception_ros(nh);

    ros::spin();
    return 0;


}