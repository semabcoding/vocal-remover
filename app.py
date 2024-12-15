from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydub import AudioSegment
from spleeter.separator import Separator
import librosa
import librosa.display
import noisereduce as nr
import soundfile as sf
import os
import re
import yt_dlp
from mimetypes import guess_type
import subprocess
import numpy as np
import threading

app = Flask(__name__)
CORS(app)

# Dossiers pour stocker les fichiers
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def extract_melody(file_path):
    y, sr = librosa.load(file_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notes = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            notes.append(librosa.hz_to_note(pitch))

    return notes[:100]  # Limite le nombre de notes extraites
def sanitize_filename(filename):
    """Nettoie le nom de fichier pour éviter les conflits."""
    return re.sub(r'[^\w\-_\.]', '_', filename)

# Convertir les fréquences en notes Do Ré Mi Fa Sol La Si
def hz_to_note_name(pitch_hz):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if pitch_hz > 0:
        note_number = int(12 * np.log2(pitch_hz / 440.0) + 49)
        note_name = notes[note_number % 12]
        return note_name
    return None

def log_error(e):
    """Affiche les erreurs dans la console."""
    print(f"Error: {str(e)}")


@app.route('/vocal-remover', methods=['POST'])
def vocal_remover():
    """Sépare les éléments vocaux et musicaux d'un fichier audio."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        file = request.files['file']
        original_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(file_path)

        # Utilisation du modèle spleeter:2stems (Vocals + Accompaniment uniquement)
        separator = Separator('spleeter:2stems')
        separator.separate_to_file(file_path, PROCESSED_FOLDER)

        processed_folder_path = os.path.join(PROCESSED_FOLDER, os.path.splitext(original_filename)[0])
        processed_files = os.listdir(processed_folder_path)

        # Assurez-vous de filtrer les fichiers pour ne garder que vocals.wav et accompaniment.wav
        processed_files = [f for f in processed_files if f in ['vocals.wav', 'accompaniment.wav']]

        processed_urls = [
            f"http://127.0.0.1:5000/download/{os.path.splitext(original_filename)[0]}/{file}"
            for file in processed_files
        ]

        return jsonify({'processed_files': processed_urls}), 200
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 500

#spleeter
@app.route('/splitter', methods=['POST'])
def splitter():
    """Sépare un fichier audio en plusieurs stems."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        file = request.files['file']
        separation_type = request.form.get('separation_type', '2stems')  # Par défaut : 2 stems
        if separation_type not in ['2stems', '4stems', '5stems']:
            return jsonify({'error': 'Invalid separation type'}), 400

        original_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(file_path)

        # Configuration de spleeter pour le type de séparation choisi
        separator = Separator(f"spleeter:{separation_type}")
        separator.separate_to_file(file_path, PROCESSED_FOLDER)

        processed_folder_path = os.path.join(PROCESSED_FOLDER, os.path.splitext(original_filename)[0])
        processed_files = os.listdir(processed_folder_path)

        # Filtrage des fichiers en fonction du type de séparation
        if separation_type == '2stems':
            processed_files = [f for f in processed_files if f in ['vocals.wav', 'accompaniment.wav']]
        elif separation_type == '4stems':
            processed_files = [f for f in processed_files if f in ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']]
        elif separation_type == '5stems':
            processed_files = [f for f in processed_files if f in ['vocals.wav', 'drums.wav', 'bass.wav', 'piano.wav', 'other.wav']]

        processed_urls = [
            f"http://127.0.0.1:5000/download/{os.path.splitext(original_filename)[0]}/{file}"
            for file in processed_files
        ]

        return jsonify({'processed_files': processed_urls}), 200
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 500





### Pitch Changer ###
@app.route('/pitch', methods=['POST'])
def change_pitch():
    """Change le pitch d'un fichier audio."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        file = request.files['file']
        pitch_value = float(request.form.get('pitch', 0))
        original_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(file_path)

        audio = AudioSegment.from_file(file_path)
        new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_value / 12.0)))
        shifted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
        processed_audio = shifted_audio.set_frame_rate(audio.frame_rate)

        output_path = os.path.join(PROCESSED_FOLDER, "pitch")
        os.makedirs(output_path, exist_ok=True)
        processed_file_path = os.path.join(output_path, original_filename)
        processed_audio.export(processed_file_path, format="wav")

        processed_url = f"http://127.0.0.1:5000/download/pitch/{original_filename}"
        return jsonify({'processed_files': [processed_url]}), 200
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 500


### Key & BPM Finder ###
@app.route('/key-bpm', methods=['POST'])
def key_bpm_finder():
    """Détecte la clé musicale et le BPM d'un fichier audio."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        file = request.files['file']
        original_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(file_path)

        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_index = chroma.mean(axis=1).argmax()
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]

        return jsonify({'key': key, 'bpm': round(tempo, 2)}), 200
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 500


### Audio Cutter ###
@app.route('/audio-cutter', methods=['POST'])
def audio_cutter():
    """Cuts the selected portion of an audio file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        start_time = float(request.form.get('start', 0))
        end_time = float(request.form.get('end', 0))
        filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        audio = AudioSegment.from_file(file_path)
        cut_audio = audio[start_time * 1000:end_time * 1000]

        output_path = os.path.join(PROCESSED_FOLDER, f"cut_{filename}")
        cut_audio.export(output_path, format="wav")

        processed_url = f"http://127.0.0.1:5000/download/{os.path.basename(output_path)}"
        return jsonify({'processed_file': processed_url}), 200
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 500





### Noise Reducer ###
@app.route('/remove-noise', methods=['POST'])
def remove_noise():
    """Supprime le bruit d'un fichier audio."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        file = request.files['file']
        noise_percentage = float(request.form.get('percentage', 100))
        original_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(file_path)

        y, sr = librosa.load(file_path, sr=None)
        noise_sample = y[:int(sr * 0.5)]
        reduced_noise = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=noise_percentage / 100.0)

        output_filename = f"denoised_{original_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        sf.write(output_path, reduced_noise, sr)

        processed_url = f"http://127.0.0.1:5000/download/{output_filename}"
        return jsonify({'processed_file': processed_url}), 200
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 500
    

    #melody
@app.route('/melody', methods=['POST'])
def melody_finder():
    try:
        # Vérifie si un fichier ou un lien YouTube est fourni
        if 'file' in request.files:
            file = request.files['file']
            filename = sanitize_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
        elif 'url' in request.form:
            youtube_url = request.form.get('url')
            filename = "downloaded_audio.mp3"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            command = ["yt-dlp", "--extract-audio", "--audio-format", "mp3", "--output", file_path, youtube_url]
            subprocess.run(command, check=True)
        else:
            return jsonify({'error': 'No file or URL provided'}), 400

        # Charger l'audio
        y, sr = librosa.load(file_path, sr=None)

        # Extraire les notes et les timestamps
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        times = librosa.frames_to_time(onset_frames, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        melody_data = []

        for time, frame in zip(times, onset_frames):
            index = magnitudes[:, frame].argmax()
            pitch = pitches[index, frame]
            if pitch > 0:
                note = librosa.hz_to_note(pitch)
                melody_data.append({'time': round(time, 2), 'note': note})

        return jsonify({'melody': melody_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
### Download Route ###
@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Télécharge un fichier traité."""
    try:
        return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)
    except Exception as e:
        log_error(e)
        return jsonify({'error': str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True)
