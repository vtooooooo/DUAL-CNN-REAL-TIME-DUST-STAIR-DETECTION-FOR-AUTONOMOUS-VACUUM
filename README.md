# Smart-Autonomous Vacuum Cleaner Using CNN and AI Navigation

## Problem Statement

Traditional robotic vacuum cleaners typically rely on hardcoded or random cleaning patterns and lack the intelligence to detect dusty areas or stairs in real time. This results in inefficient cleaning and safety risks when navigating complex environments like multi-story homes.

## Solution

This project simulates a smart vacuum cleaner that:
- Detects **dust** and **stairs** using trained **Convolutional Neural Networks (CNNs)**
- Navigates intelligently using memory-based logic to avoid re-cleaning the same areas
- Supports **multi-floor cleaning** with stair detection and simulated stair climbing
- Visualizes the entire cleaning and movement process using **OpenCV**

## What Makes It Unique?

Unlike most simulations or commercial robotic vacuum logic, **our model introduces autonomous multi-floor cleaning**:

- After fully cleaning the **ground floor**, the vacuum **detects stairs using a CNN**
- It then **climbs stairs virtually** and continues cleaning the **first floor**, then the **second**, and so on
- This behavior creates a **fully autonomous cleaning flow across multiple levels**

## Data

- **Dust Detection Dataset**: Images of dusty and clean wooden/marble floors
- **Stair Detection Dataset**: Images labeled as **Stair**, **Plain Floor**, **Obstacle**, or **Unknown**
- Data was **augmented** using `preprocess_augment.py` to increase variety and accuracy

## Models

Two TensorFlow-based CNN models were trained:
- `dust_detection_cnn.h5`: Classifies patches as **Dusty** or **Clean**
- `stair_detection_cnn.h5`: Classifies patches as **Stair**, **Plain Floor**, **Obstacle**, or **Unknown**

These models are used during simulation for real-time classification and decision-making.

## Features

- Intelligent movement logic with memory of visited locations
- Real-time prediction using deep learning models
- Virtual stair detection and climbing animation
- Multi-floor cleaning simulation using dynamic map generation

## File Structure

| File | Description |
|------|-------------|
| `integration.py` | Main simulation script with movement logic and model inference |
| `testing.py` | Alternate version for simulation and testing |
| `preprocess_augment.py` | Script to resize and augment image data |
| `dust_detection_cnn.h5` | Trained CNN for dust detection |
| `stair_detection_cnn.h5` | Trained CNN for stair/floor classification |

## How to Run

1. Clone the repo
   git clone https://github.com/Yogananda16/smart-vacuum-cleaner-ai.git
   cd smart-vacuum-cleaner-ai
2. Install dependencies:
   pip install tensorflow opencv-python pillow tqdm numpy
3. Python integration.py
4. Press ESC to close the simulation window

## Tech Stack

1. Python
2. TensorFlow / Keras – For building and loading CNN models
3. OpenCV – For map drawing and visual simulation
4. Pillow (PIL) – For image preprocessing and augmentation
5. NumPy – For matrix and patch handling

## Future Enhancements

- Hardware Integration: Deploy the model on a physical robot using Raspberry Pi, camera, and motor drivers

- Add YOLO-based object detection to identify obstacles and furniture

- Improve navigation logic with A* pathfinding or Reinforcement Learning

- Build a mobile/web dashboard to control or schedule cleaning

- Add features like auto-charging, battery monitoring, and smart resume
