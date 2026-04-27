SHELL := /bin/bash
WS    := $(shell pwd)
TC    ?= circle

.PHONY: build sim robot clean \
        sim-circle sim-stationary sim-square sim-oscillate sim-zigzag

build:
	source /opt/ros/humble/setup.bash && \
	colcon build --symlink-install --packages-select follow

sim:
	source /opt/ros/humble/setup.bash && \
	source $(WS)/install/setup.bash && \
	CYCLONEDDS_URI=$(WS)/setup/cyclonedds.xml \
	ros2 launch follow sim.launch.py test_case:=$(TC)

sim-circle:
	$(MAKE) sim TC=circle

sim-stationary:
	$(MAKE) sim TC=stationary

sim-square:
	$(MAKE) sim TC=square

sim-oscillate:
	$(MAKE) sim TC=oscillate

sim-zigzag:
	$(MAKE) sim TC=zigzag

robot:
	source /opt/ros/humble/setup.bash && \
	source $(WS)/install/setup.bash && \
	CYCLONEDDS_URI=$(WS)/setup/cyclonedds.xml \
	ros2 launch follow robot.launch.py

clean:
	rm -rf build install log
