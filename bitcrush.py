import numpy as np
from scipy.io import wavfile
import pywt

def generate_square_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    square_wave = np.sign(np.sin(2 * np.pi * freq * t))
    return square_wave

def bitcrush(audio, bits):
    max_value = 2 ** (bits - 1) - 1
    return np.round(audio * max_value) / max_value

def wavelet_compress(audio, level):
    coeffs = pywt.wavedec(audio, 'haar', level=level)
    coeffs[1:] = [np.zeros_like(coeff) for coeff in coeffs[1:]]
    return pywt.waverec(coeffs, 'haar')

def main(input_file, output_file, bitcrush_bits, square_wave_freq):
    # Read the stereo audio file
    sample_rate, stereo_data = wavfile.read(input_file)
    
    # Separate left and right channels
    left_channel = stereo_data[:, 0]
    right_channel = stereo_data[:, 1]
    
    # Apply bitcrushing to each channel
    bitcrushed_left = bitcrush(left_channel.astype(float), bitcrush_bits)
    bitcrushed_right = bitcrush(right_channel.astype(float), bitcrush_bits)
    
    # Generate square wave matching input audio frequency
    square_wave_duration = len(bitcrushed_left) / sample_rate
    square_wave = generate_square_wave(square_wave_freq, square_wave_duration, sample_rate)
    
    # Normalize square wave amplitude
    square_wave = square_wave / np.max(np.abs(square_wave))
    
    # Streaming process for left channel
    chunk_size = 8192
    compressed_left = []
    
    for i in range(0, len(bitcrushed_left), chunk_size):
        chunk = bitcrushed_left[i: i + chunk_size]
        chunk_square_wave = square_wave[i: i + chunk_size]
        mixed_chunk = chunk * chunk_square_wave
        compressed_chunk = wavelet_compress(mixed_chunk, wavelet_level)
        compressed_left.extend(compressed_chunk)
    
    compressed_left = np.array(compressed_left)
    
    # Streaming process for right channel (similar to left channel)
    compressed_right = []
    
    for i in range(0, len(bitcrushed_right), chunk_size):
        chunk = bitcrushed_right[i: i + chunk_size]
        chunk_square_wave = square_wave[i: i + chunk_size]
        mixed_chunk = chunk * chunk_square_wave
        compressed_chunk = wavelet_compress(mixed_chunk, wavelet_level)
        compressed_right.extend(compressed_chunk)
    
    compressed_right = np.array(compressed_right)
    
    # Combine left and right channels
    compressed_stereo = np.column_stack((compressed_left, compressed_right))
    
    # Normalize the audio to the original range
    compressed_stereo = np.interp(compressed_stereo, (compressed_stereo.min(), compressed_stereo.max()), (-1, 1))
    
    # Save the compressed stereo audio
    wavfile.write(output_file, sample_rate, compressed_stereo.astype(np.float32))

if __name__ == "__main__":
    input_file = "voice.wav"  # Replace with your input stereo audio file
    output_file = "output.wav"  # Replace with your desired output file
    bitcrush_bits = 32  # Number of bits for bitcrushing
    square_wave_freq = 440  # Frequency of the square wave (adjust as desired)
    wavelet_level = 2  # Wavelet compression level
    
    main(input_file, output_file, bitcrush_bits, square_wave_freq)

