import wave
import numpy as np


def decompose_wav(file_path):

    """
    Extract and separate audio data, sample rate, and channel number from given file

    :parameter file_path - the path of the audio file

    :returns audio_data - channel_count dimensional array of audio data
    :returns frame_rate - sample rate of recording
    :returns channel_count - number of channels in audio data

    """

    # Open the WAV file
    wav_file = wave.open(file_path, 'rb')

    # Extract parameters
    channel_count = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()

    # Read frames and convert to byte data
    raw_data = wav_file.readframes(n_frames)

    # Close the WAV file
    wav_file.close()

    # Convert byte data to integers based on the sample width
    if sample_width == 1:
        dtype = np.uint8  # 8-bit audio (unsigned)
    elif sample_width == 2:
        dtype = np.int16  # 16-bit audio (signed)
    elif sample_width == 4:
        dtype = np.int32  # 32-bit audio (signed)
    else:
        raise ValueError("Unsupported sample width")

    # Convert the raw byte data to numpy array
    audio_data = np.frombuffer(raw_data, dtype=dtype)

    mono_audio_data = np.mean(audio_data, axis=1)
    mono_audio_data = mono_audio_data.astype(audio_data.dtype)

    # Normalize the data
    # audio_data = audio_data / np.max(np.abs(audio_data), axis=0)

    return mono_audio_data, frame_rate, channel_count


