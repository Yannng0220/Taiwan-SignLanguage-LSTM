# Taiwan-SignLanguage-LSTM
I used LSTM to train Taiwanese Sign Language, and used Mediapipe to draw body nodes, used Tensorflow with the keras suite to train the model, and used numpy to draw key points.

To install the required packages (Mediapipe, Tensorflow, Keras, numpy, matplotlib), you can use the following commands in your Python environment. Make sure you have Python installed before proceeding:

［pip install mediapipe tensorflow keras numpy matplotlib］
After installing these packages, you can proceed with the tasks you mentioned:

Open the "openh5.py" file.
Execute the code to train the predefined Taiwanese sign language models.
To add new signs, use "LSTMCOLLECT.py" for data collection. After collecting the data, proceed to train the model using "Trainh5.py" as mentioned below.
In "Trainh5.py," you can modify the following parameters:
no.sequences: The number of sequences or sets of images.
sequence_length: The number of images per sequence.
You can also modify the number of LSTM layers in the code.
Make sure to adjust these parameters according to your specific requirements.
