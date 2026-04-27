FROM althack/ros2:humble-dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    tmux \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

# Python dependencies (torch, SB3, LSTM, plotting)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --timeout 2000 -r /tmp/requirements.txt

# Auto-source workspace on container start
ARG WORKSPACE
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc

# CycloneDDS and ROS env
RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/ros/.bashrc
RUN echo "export CYCLONEDDS_URI=${WORKSPACE}/setup/cyclonedds.xml" >> /home/ros/.bashrc
