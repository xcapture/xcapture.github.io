import glob
import os
import tempfile

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import soundfile as sf
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

# High-pass filter function
def high_pass_filter(data, sr, cutoff=100, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Spectrogram plotting function
def plot_spec(sample, sr=44100, n_fft=256, hop_length=64, vmin=None, vmax=None, ax=None):
    if len(sample.shape) > 1 and sample.shape[1] > 1:  # Use first channel for stereo
        sample = sample[:, 0]

    D = librosa.stft(sample, hop_length=hop_length, n_fft=n_fft)  # STFT
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, n_fft=n_fft,
                                   x_axis=None, y_axis=None, ax=ax,
                                   vmin=vmin, vmax=vmax, cmap="magma")
    return img, S_db

# Function to create a video with a moving playhead
def create_spectrogram_video(audio_file, output_video, n_fft=256, hop_length=64, begin_offset=None, n_samples=None):
    # Load the audio
    y, sr = librosa.load(audio_file, mono=False, sr=None)
    if len(y.shape) > 1:
        y = y[0, ...]
    if begin_offset:
        y = y[begin_offset:]
    if n_samples:
        y = y[:n_samples]
    y = high_pass_filter(y, sr)
    y = y / np.max(np.abs(y)) * 0.99

    # Save the processed audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
        processed_audio_file = tmp_audio_file.name
        sf.write(processed_audio_file, y, sr)

    # Compute spectrogram
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)  # Use tight_layout
    ax.axis("off")  # Turn off axes
    img, S_db = plot_spec(y, sr, n_fft=n_fft, hop_length=hop_length, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Convert spectrogram to a NumPy image
    spec_image = mplfig_to_npimage(fig)
    plt.close(fig)

    # Parameters for video
    duration = librosa.get_duration(y=y, sr=sr)
    frames_per_second = 30
    num_frames = int(duration * frames_per_second)

    # Function to create a single frame with a moving playhead
    def make_frame(t):
        # frame_fig, frame_ax = plt.subplots(figsize=(10, 6))
        # frame_ax.axis("off")

         # Create a figure
        frame_fig = plt.figure(figsize=(8, 6))
        
        # Add an axis that occupies the full figure space
        frame_ax = frame_fig.add_axes([0, 0, 1, 1])  # Full figure space, no margins
        frame_ax.axis("off")  # Turn off axes

        img, _ = plot_spec(y, sr, n_fft=n_fft, hop_length=hop_length, ax=frame_ax)
        x_max = S_db.shape[1]  # Total number of frames in the spectrogram
        num_frames = frames_per_second * duration
        x_pos = int((t / duration) * (x_max / (num_frames / (num_frames+1))))  # Ensure playhead reaches the last frame


        # Plot playhead
        frame_ax.axvline(x=x_pos, color="red", linewidth=2)
        frame_image = mplfig_to_npimage(frame_fig)
        plt.close(frame_fig)
        return frame_image

    # Create video
    video = mpy.VideoClip(make_frame, duration=duration)
    audio = mpy.AudioFileClip(processed_audio_file)  # Load audio
    video = video.set_audio(audio)  # Add audio to the video
    video.write_videofile(output_video, fps=frames_per_second, codec="libx264", audio_codec="aac")
    print(f"Video saved to {output_video}")

# Main script
def main():
    # Directories
    input_files = glob.glob('../examples/generation/*/input_audio.wav')
    begin_offset = 24000
    n_samples = int(48000 * 3)

    # Process each audio file
    for file_name in input_files:
        print(file_name)
        
        # Skip non-audio files
        if not os.path.isfile(file_name) or not file_name.lower().endswith(('.wav', '.mp3', '.flac')):
            continue

        try:
            # # Load the audio file
            # y, sr = librosa.load(input_file, sr=None)
            # y = y[12000:200000]
            # y = high_pass_filter(y, sr)

            # # Generate spectrogram PDF
            # fig = plt.figure(figsize=(10, 6))
            # ax = fig.add_axes([0, 0, 1, 1])  # Full figure space
            # ax.axis("off")  # Remove axes
            
            # # Plot the spectrogram
            # plot_spec(y, sr=sr, ax=ax)
            # fig.set_size_inches(6, 5)

            # # Save spectrogram as PDF
            # output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".pdf")
            # plt.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0)
            # plt.close(fig)
            # print(f"Spectrogram saved as PDF in {output_file}")

            # Generate spectrogram video

            output_video = os.path.splitext(file_name)[0] + ".mp4"
            print(output_video)
            create_spectrogram_video(file_name, output_video, begin_offset=begin_offset, n_samples=n_samples)
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

if __name__ == "__main__":
    main()
