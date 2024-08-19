from decompose_wav import decompose_wav
import numpy as np
import matplotlib.pyplot as plt

target_file = "untitled.wav"

# channel count is necessary to know dimensionality of audio data
# sample rate is for calculating limits and sample size for fourier transform function
audio_data, sample_rate, channel_count = decompose_wav(target_file)

# run through every frequency once: every point gets analyzed for same frequency at one a time
sample_size = 1024  # arbitrarily small to prevent modulation from voice form butting in
start_point = 0
fft_result = np.fft.fft(audio_data[start_point:start_point + sample_size])
# generates an array of length sample_size

# Compute the corresponding frequency bins
frequencies = np.fft.fftfreq(len(audio_data[start_point:start_point + sample_size]), 1 / sample_rate)
frequencies = frequencies[:len(frequencies) // 2]
# //2 to get rid of negative side of graph, as it is a mirror of the positive side

# Compute the magnitude of the FFT result
magnitude = np.abs(fft_result)
magnitude = magnitude[:len(magnitude) // 2]
# //2 for magnitudes in order to keep the point values even (as half of the graph now has a magnitude but no freq

# Plot the positive frequencies and their magnitudes
plt.figure(figsize=(10, 6))

plt.plot(frequencies, magnitude)
plt.title("FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# Find the peak frequency component
peak_index = np.argmax(magnitude)  # Find the index of the peak magnitude
peak_frequency = frequencies[peak_index]
peak_magnitude = magnitude[peak_index]
print(f"The dominant frequency component is {peak_frequency} Hz with a magnitude of {peak_magnitude}.")

# np.argsort() is weird
# np.argsort(array) returns the indices that would sort the array
# slicing -N+1 gets the bottom (largest) N+1 items
# slicing 1: removes the top item, as we already know the peak

N = 3  # number of top values to extract (minus peak)
top_n_indices = np.argsort(magnitude[:len(magnitude) // 2])[-N + 1:][::-1][1:]
for index in top_n_indices:
    if magnitude[index] / peak_magnitude >= 0.4:  # assuming that 40% of the peak is still a relevant "peak" frequency
        print(
            f"Another prominent frequency is {frequencies[index]} Hz "
            f"with a relative magnitude (to peak) of {magnitude[index] / peak_magnitude}."
        )
