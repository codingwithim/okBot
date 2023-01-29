# Chatbot System for Mental Health in Bahasa Malaysia
This chatbot system is integrated with artificial intelligence and natural language processing. This chatbot utilize the feedforward neural network model to train the datasets. Kivy and KivyMD are used to create the app's graphical user interface.

![image](https://user-images.githubusercontent.com/111273105/184799065-1c126db6-6e56-4dac-8fb0-cec16be2c610.png)
## Requirements
To run this project, you will need the following:

    Python 3.6 or above
    kivy 2.0.0
    kivymd=0.104.1
    TensorFlow 2.4.0 or above
    NLTK 3.6.7
    Numpy 1.18.5 or above
    Pandas 1.0.4 or above
You can install these packages by running the following command:

    conda install --file requirements.txt

## Usage

1. Download or clone the repository to your local machine
2. Install the required packages
3. Open a terminal and navigate to the project directory.
4. Run the following command to start the chatbot:

        python main.py

## Directory Structure
* Fonts: contains the fonts used in the project
* Images: contains the images used in the project
* JSON_FILES: contains the intents.json file used for training the model
* Kivy Files: contains the kivy files for building the chatbot interface
* pickle_files: contains the words.pkl and classes.pkl files used for loading the model
* Results: contains the results of the chatbot's performance
* training.py: script for training the model using neural network
* main.py: script for running the chatbot
chatbotmodel.h5: trained model file

## References

1) The research paper for this project can be found on [ResearchGate](https://www.researchgate.net/publication/364145271_Chatbot_System_for_Mental_Health_in_Bahasa_Malaysia)
2) Watch the demo of the chatbot here [Youtube](https://youtu.be/FfNAC5Gsg80)

## Disclaimer
This project utilizes state-of-the-art natural language processing and neural network techniques to provide a mental health chatbot in Bahasa Malaysia. However, please note that the accuracy of the chatbot may be limited due to the lack of a large amount of conversational data in Bahasa Malaysia. Nevertheless, we believe that the chatbot can be improved by providing more datasets for training. Your contributions in terms of data or feedback would be highly appreciated.







