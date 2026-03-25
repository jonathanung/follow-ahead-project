# follow-ahead-project

ROS 2 Humble project with Turtlebot3 simulation and reinforcement learning.

## Docker

### Prerequisites
- Docker
- Docker Compose

### Build & Run

```bash
# Build the image
docker compose build

# Start the container
docker compose up -d

# Attach to container
docker compose exec follow-ahead bash
```

### Useful Commands

```bash
# View logs
docker compose logs -f

# Stop the container
docker compose down
```

## Development

The project uses Python 3.10 with ROS 2 Humble. Key dependencies:
- PyTorch 2.3.1 (CPU)
- Stable-Baselines3 2.7.1
- Gymnasium 1.0.0
- SciPy, Matplotlib, NumPy, CasADi
