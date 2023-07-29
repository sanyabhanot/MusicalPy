# Abstract

Over the past several years, deep learning for sequence modeling has grown in popularity. To achieve this goal, LSTM network structures have proven to be very useful for making predictions for the next output in a series. For instance, a smartphone predicting the next word of a text message could use an LSTM. We sought to demonstrate an approach of music generation using Recurrent Neural Networks (RNN). More specifically, a Long Short-Term Memory (LSTM) neural network. Generating music is a notoriously complicated task, whether handmade or generated, as there are a myriad of components involved. Taking this into account, we provide an application of LSTMs in music generation, develop and present the network we found to best achieve this goal, identify and address issues and challenges faced, and include potential future improvements for our network.

# Introduction

Machine learning is now being used for many interesting applications in a variety of fields. Music generation is one of the interesting applications of machine learning. As music itself is sequential data, it can be modelled using a sequential machine learning model such as the recurrent neural network. This modelling can help in learning the music sequence and generating the sequence of music data.
**Automatic Music Generation** is a process where a system, given a sequence of notes, will learn to predict the next note in the sequence. 
In this Project we are going to use the Piano Instrument with the following note variables:
* Pitch
* Step
* Duration

### The Maestro Dataset
MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) is a dataset composed of about 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms. Tensorflow partnered with organizers of
the International Piano-e-Competition for the raw data used in this dataset. The dataset contains about 200 hours of paired audio and MIDI recordings from ten years of International Piano-e-Competition. Audio and MIDI files are aligned with ∼3 ms accuracy and sliced to individual musical pieces, which are annotated with composer, title, and year of performance.

### Libraries 
* Numpy
* Pandas
* Tensorflow
* Matplotlib
* Seaborn
* PyFluidSynth
* FluidSynth
* Pretty_midi

# Methodology

### 1. Description of Method
We are using the LSTM (Long Short Term Memory) network for this project. LSTM is a type of RNN that learns efficiently via gradient descent and they recognize and encode long-term patterns using gating mechanisms. In cases where a network has to remember information for a long period of time (like music and text) LSTM is extremely useful.
![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/c34c9bf3-49a0-4c0c-82bc-b92973f0a5b8)

### 2. Data Gathering
We train a model using a collection of piano MIDI files from the MAESTRO dataset. Given a sequence of notes, our model will learn to predict the next note in the sequence. We can generate a longer sequence of notes by calling the model repeatedly. The MAESTRO dataset
contains about 1,200 MIDI files.

import pathlib
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
tf.keras.utils.get_file(
'maestro-v2.0.0-midi.zip',
origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/ma,extract=True,
cache_dir='.', cache_subdir='data')

### 2. Data Preparation
We used the pretty_midi library to create and parse MIDI files, and pyfluidsynth for generating audio playback in Colab. First, use pretty_midi to parse a single MIDI file and inspect the format of the notes. We generate a PrettyMIDI object for the sample MIDI file.
We load the data as an array using get_note_name() function and the pseudo code is given below:
For all midi files:
1. Get notes using PrettyMIDI
2. Add the notes to the list

raw_notes = midinotes(first_file)
get_note_names = np.vectorize(pretty_midi.note_number_to_name)
first_note_names = get_note_names(raw_notes['pitch'])

### The Models
We have used 4 types of layers in our model:
1. Input Layer
2. LSTM Layer - 1 layer
3. Dense Layer -3 layers
4. Output Layer

![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/367938c7-f001-4122-911e-93342b4d64e4)

Loss is calculated using sparse categorical cross entropy as our outputs belong to a single class and we have more than two classes to work with. Adam optimizer is used as an optimizer to optimize our LSTM. The network is trained using 50 epochs with each batch size of 64. Model checkpoints are used to save the intermediate models and generate output using them.

![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/265e966e-9afb-4699-9fde-166f6bfae8de)

# Results

### Recognition of the Instrument
Our model successfully identifies the number and the type of instrument being played in the MIDI files. It also shows the starting, ending, pitch and the velocity of the extracted note.
![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/4296744b-391e-4b86-b20c-b4e0b9700337)

### Analysing the Loss
We plotted the graph between the loss and the epochs to get a closer look at the total loss experienced while getting the output.
![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/6c14f6a4-44f2-4427-8f4b-69b87082f9eb)

### Generating the Notes
To use the model to generate notes, we first need to provide a starting sequence of notes. For note pitch, it draws a sample from the softmax distribution of notes produced by the model,and does not simply pick the note with the highest probability. Always picking the
note with the highest probability would lead to repetitive sequences of notes being generated.
The generated notes function produces a table showing the starting and ending of the three parameters: pitch, step and duration. We can change the temperature and the starting sequence in next_notes to produce different outputs.

![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/c2643f41-cb0b-45c6-8838-7c7af1e0920e)

### Plots and Graphs
![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/2a9bfe1f-c89c-4803-94a3-85c4e6e743c8)

We extracted the generated notes in the form of a sheet just like a piano sheet
![image](https://github.com/sanyabhanot/MusicalPy/assets/111521883/44264dec-e464-47dc-b95b-1ccfd7aa4586)

# Conclusion

With this project, we have demonstrated how to create a LSTM neural network to generate music. While the results may not be perfect, they are impressive nonetheless and shows us that neural networks can create music and could potentially be used to help create more
complex musical pieces.
With potentially more complex models, there is a lot neural networks can do in the field of creative work. While AI is no replacement for human creativity, it’s easy to see a collaborative future between artists, data scientists, programmers, and researchers to create
more compelling content.

# Future Scope

We have achieved remarkable results and beautiful melodies by using a simple LSTM network. However, there are areas that can be improved.

• Add beginnings and endings to pieces. As there is no distinction between pieces, the network does not know where one piece ends and another one begins. This would allow the network to generate a piece from start to finish instead of ending the generated piece abruptly
as it does now.

• Adding more instruments to the dataset. As it is now, the network only supports pieces that only have a single instrument. It would be interesting to see if it could be expanded to support a whole orchestra.






