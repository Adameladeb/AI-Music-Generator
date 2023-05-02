import os
import numpy as np
from music21 import midi, converter, instrument, note, chord, meter, key
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical
from flask import Flask, render_template, request, send_file
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QComboBox
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Initialize the Flask app
app = Flask(__name__)

class MidiGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('AI Music Generator')
        self.setWindowIcon(QIcon('icon.png'))
        self.resize(500, 300)
        self.center()

        self.filepath = None

        # Load MIDI file button
        self.load_button = QPushButton('Load MIDI file', self)
        self.load_button.clicked.connect(self.load_file)
        self.load_button.resize(120, 40)
        self.load_button.move(20, 20)

        # Sequence length label and input box
        self.seq_length_label = QLabel('Sequence length:', self)
        self.seq_length_label.move(20, 80)
        self.seq_length_box = QLineEdit(self)
        self.seq_length_box.setText('100')
        self.seq_length_box.resize(40, 25)
        self.seq_length_box.move(120, 80)

        # Generate button
        self.gen_button = QPushButton('Generate', self)
        self.gen_button.clicked.connect(self.generate)
        self.gen_button.resize(120, 40)
        self.gen_button.move(20, 140)

        # Temperature label and input box
        self.temp_label = QLabel('Temperature:', self)
        self.temp_label.move(20, 200)
        self.temp_box = QLineEdit(self)
        self.temp_box.setText('0.5')
        self.temp_box.resize(40, 25)
        self.temp_box.move(120, 200)

        # Output file format selection box
        self.format_label = QLabel('Output format:', self)
        self.format_label.move(250, 80)
        self.format_box = QComboBox(self)
        self.format_box.addItem('MIDI')
        self.format_box.addItem('MP3')
        self.format_box.resize(100, 25)
        self.format_box.move(350, 80)

        # Output file name input box
        self.name_label = QLabel('Output file name:', self)
        self.name_label.move(250, 140)
        self.name_box = QLineEdit(self)
        self.name_box.setText('output')
        self.name_box.resize(150, 25)
        self.name_box.move(350, 140)

        # Generate button
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_file)
        self.save_button.resize(120, 40)
        self.save_button.move(250, 200)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_file(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, 'Load MIDI file', '', 'MIDI files (*.mid *.midi);;All files (*.*)')
        if self.filepath:
            QMessageBox.information(self, 'File loaded', 'MIDI file loaded successfully.')

    def generate(self):
    if not self.filepath:
        QMessageBox.warning(self, 'Error', 'Please load a MIDI file first.')
        return

    # Load MIDI file and extract notes/chords
    midi_stream = converter.parse(self.filepath)
    notes = []
    for part in midi_stream.parts:
        instr = instrument.partitionByInstrument(part)
        if instr:
            notes += [n for n in instr.parts[0].recurse() if isinstance(n, note.Note) or isinstance(n, chord.Chord)]
        else:
            notes += [n for n in part.recurse() if isinstance(n, note.Note) or isinstance(n, chord.Chord)]

    # Build vocabulary of unique notes/chords
    pitchnames = sorted(set(item.nameWithOctave for item in notes))
    note_to_int = dict((note, num) for num, note in enumerate(pitchnames))
    int_to_note = dict((num, note) for num, note in enumerate(pitchnames))

    # Prepare training sequences
    seq_length = int(self.seq_length_box.text())
    input_sequences = []
    output_sequences = []
    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        input_sequences.append([note_to_int[str(n)] for n in seq_in])
        output_sequences.append(note_to_int[str(seq_out)])
    n_patterns = len(input_sequences)

    # Reshape input sequences
    X = np.reshape(input_sequences, (n_patterns, seq_length, 1))
    X = X / float(len(pitchnames))

    # One-hot encode output sequences
    y = to_categorical(output_sequences)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(len(pitchnames)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load weights from pre-trained model
    model.load_weights('weights.hdf5')

    # Generate new sequence
    start = np.random.randint(0, len(input_sequences) - 1)
    pattern = input_sequences[start]
    generated_notes = []
    for i in range(500):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(pitchnames))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        generated_notes.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # Create new MIDI file from generated sequence
    output_filetype = self.format_box.currentText().lower()
    if output_filetype == 'midi':
        output_filename = self.name_box.text() + '.mid'
    else:
        output_filename = self.name_box.text() + '.mp3'
    output_filepath = os.path.join(os.getcwd(), output_filename)
    offset = 0
    output_notes = []
    for pattern in generated_notes:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            offset += 0.5
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write(output_filetype, output_filepath)

    QMessageBox.information(self, 'Success', 'MIDI file generated successfully.')


