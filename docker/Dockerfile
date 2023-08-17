# syntax=docker/dockerfile:1

FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"] 

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs

# https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md
# Finally, create a ROS 2 workspace for experimenting with Isaac ROS:
RUN mkdir -p  ~/workspaces/isaac_ros-dev/src
RUN echo "export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev/" >> ~/.bashrc
RUN echo "export TERM='xterm-256color'" >> ~/.bashrc
RUN source ~/.bashrc

RUN echo $TERM

# https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection
# Source ROS2
RUN ./ros_entrypoint.sh
RUN printenv | grep -i ROS

# Clone this repository and its dependencies under ~/workspaces/isaac_ros-dev/src.
RUN git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection ~/workspaces/isaac_ros-dev/src/isaac_ros_object_detection
RUN git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference ~/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference
RUN git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline ~/workspaces/isaac_ros-dev/src/isaac_ros_image_pipeline
RUN git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros ~/workspaces/isaac_ros-dev/src/isaac_ros_nitros
RUN git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common ~/workspaces/isaac_ros-dev/src/isaac_ros_common

# Pull down a ROS Bag of sample data:
RUN cd ~/workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet && \
  git lfs pull -X "" -I "resources/rosbags"

# Launch the Docker container using the run_dev.sh script:
RUN cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
  ./scripts/run_dev.sh

# Inside the container, build and source the workspace:
RUN cd /workspaces/isaac_ros-dev && \
  colcon build --symlink-install && \
  source install/setup.bash

# (Optional) Run tests to verify complete and correct installation:
RUN colcon test --executor sequential

# Run the quickstart setup script which will download the PeopleNet Model from NVIDIA GPU Cloud(NGC)
RUN cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet && \
  ./scripts/setup_model.sh --height 632 --width 1200 --config-file resources/quickstart_config.pbtxt

# Run the following launch file to spin up a demo of this package:
# RUN cd /workspaces/isaac_ros-dev && \
#   ros2 launch isaac_ros_detectnet isaac_ros_detectnet_quickstart.launch.py