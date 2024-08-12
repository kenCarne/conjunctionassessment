# conjunctionassessment

# Satellite Collision Risk Visualization

## Overview

This project provides a Python-based tool to visualize and analyze satellite collision risks using Two-Line Element sets (TLEs). It features:

- **3D Visualization**: Displays satellite orbits, Earth, and potential collision hotspots.
- **Collision Probability Calculation**: Estimates the probability of collisions between satellites.
- **Heatmap of Collision Hotspots**: Highlights areas in space where collisions are most likely to occur.
- **Multi-Processing**: Utilizes multi-processing to handle large datasets efficiently.

## Features

- **Satellite Orbits Visualization**: Plot orbits of selected satellites and their conjunction satellites if a collision risk is detected.
- **Collision Hotspots**: Generate a heatmap indicating regions with a high probability of collision.
- **Customizable Thresholds**: Adjust the collision probability threshold to filter conjunctions.
- **Multi-Processing**: Speed up calculations using multi-processing.

## Installation

### Prerequisites

- Python 3.7 or higher
- Install the required Python packages:

```bash
pip install numpy scipy sgp4 plotly
