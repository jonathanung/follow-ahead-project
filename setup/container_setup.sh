#!/bin/bash
# Run once when the devcontainer is created (postCreateCommand).
# Builds the follow package inside the container workspace.
set -e

# tmux quality-of-life
if [ ! -f ~/.tmux.conf ]; then
    echo "set -g history-limit 10000" >> ~/.tmux.conf
    echo "set -g mouse on" >> ~/.tmux.conf
fi

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$WS"

# Resolve ROS deps
sudo apt-get update -qq
rosdep update
rosdep install -i --from-path follow --rosdistro humble -y

# Build
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select follow

echo ""
echo "Setup complete. Open a new terminal or run:"
echo "  source install/setup.bash"
echo "Then launch with:"
echo "  make sim          # simulation (default: circle)"
echo "  make robot        # real robot (Vicon required)"
