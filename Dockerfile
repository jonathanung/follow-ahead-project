FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV TURTLEBOT3_MODEL=waffle_pi
ENV ROS_DISTRO=humble
ENV QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gazebo \
    libgazebo-dev \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-simulations \
    ros-humble-turtlebot3-msgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-ros2-controllers \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-geometry2 \
    python3-colcon-common-extensions \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt

WORKDIR /workspace
