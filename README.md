Real-Time Sign Language Recognition & Virtual Mouse
This project uses a Convolutional Neural Network (CNN) and computer vision to recognize American Sign Language (ASL) gestures from a live webcam feed. It also features a "Virtual Mouse" mode that allows you to control your computer's cursor using hand gestures.

Features
Real-Time Sign Recognition: Translates hand signs into text on the screen.

Virtual Mouse: Control your mouse cursor by moving your index finger.

Pinch to Click: Perform a mouse click by pinching your thumb and index finger together.

Customizable Dataset: Easily collect data for new signs to extend the model's vocabulary.

Text-to-Speech: Speak the recognized sentence out loud.

Project Structure
The project is divided into three core scripts:

data_collection.py: A script to capture and save images of hand signs from your webcam to build a dataset.

train_model.py: A script that trains a CNN model on the collected image data.

real_time_sign_recogniser.py: The main application that runs the live recognition and virtual mouse using the trained model.

Setup & Installation
Follow these steps to set up the project on your local machine.

1. Clone the Repository
git clone https://github.com/jaya-201/Sign-Language-and-Virtual-Mouse
cd your-repo-name

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

# Create the environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (macOS/Linux)
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

How to Run the Project
The project runs in three stages: Data Collection, Model Training, and Live Recognition.

Stage 1: Collect Data
You must collect image data for every sign you want the model to learn.

Open data_collection.py and change the label variable to the name of the sign you are recording (e.g., 'A', 'B', 'Hello').

Run the script from your terminal:

python data_collection.py

A window will appear. Position your hand to make the sign. The script will automatically save images to the dataset folder.

Repeat this process for every sign.

Stage 2: Train the Model
After collecting data for all your signs, train the model.

Run the training script from your terminal:

python train_model.py

This script will train a CNN on the images in the dataset folder and save the trained model as sign_model.h5 and the labels as categories.npy.

Stage 3: Run the Live Recognizer
With a trained model, you can now run the main application.

Run the real-time script from your terminal:

python real_time_sign_recogniser.py

A window will open with your webcam feed, and the system will start recognizing signs.

Application Controls
m: Switch between Sign Recognition and Virtual Mouse modes.

c: Clear the recognized text.

s: Speak the recognized text out loud.

q: Quit the application.
