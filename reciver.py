import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import find_peaks

chords = {
    'do_maior': [523.25, 659.25, 783.99],
    're_menor': [587.33, 698.46, 880.00],
    'mi_menor': [659.25, 783.99, 987.77],
    'fa_maior': [698.46, 880.00, 1046.50],
    'sol_maior': [783.99, 987.77, 1174.66],
    'la_menor': [880.00, 1046.50, 1318.51],
    'si_menor5b': [493.88, 587.33, 698.46]
}

def main():
    fs = 44100
    duration = 3

    print("Captando o sinal de áudio via microfone...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    recording = recording.flatten()

    # Normalização do sinal
    max_amp = np.max(np.abs(recording))
    print("Amplitude máxima capturada:", max_amp)
    if max_amp > 0:
        recording = recording / max_amp
        if max_amp < 0.1:
            recording *= 2  # Aplica ganho se o sinal for muito fraco
    else:
        print("Sinal muito fraco ou ausente!")
        return

    # Cálculo da FFT
    N = len(recording)
    fft_rec = np.fft.fft(recording)
    fft_freq = np.fft.fftfreq(N, d=1/fs)
    magnitude = np.abs(fft_rec)

    # Filtra frequências relevantes (450 Hz a 1500 Hz)
    idx_relevant = np.where((fft_freq >= 450) & (fft_freq <= 1500))
    pos_freq = fft_freq[idx_relevant]
    pos_mag = magnitude[idx_relevant]

    # Identificação dos picos
    height_threshold = np.max(pos_mag) * 0.15  # Aumentado para reduzir picos espúrios
    distance = fs // 600  # ~73 Hz, para separar picos do acorde
    peaks, properties = find_peaks(pos_mag, height=height_threshold, distance=distance)

    if len(peaks) < 3:
        print("Menos de 3 picos detectados. Tentando com limiar menor...")
        height_threshold = np.max(pos_mag) * 0.05
        peaks, properties = find_peaks(pos_mag, height=height_threshold, distance=distance)

    detected_peaks = np.round(pos_freq[peaks], 2)
    print("Picos encontrados (Hz):", detected_peaks)

    # Identificação do acorde
    detected_chord = None
    max_count = 0
    tolerance = 15  # Tolerância maior para todas as frequências
    for chord, freqs in chords.items():
        count = 0
        matched_freqs = []
        for f in freqs:
            if np.any(np.abs(pos_freq[peaks] - f) < tolerance):
                count += 1
                matched_freqs.append(f)
        print(f"Testando {chord}: {count} frequências coincidem ({matched_freqs})")
        if count >= 2 and count > max_count:
            max_count = count
            detected_chord = chord

    if detected_chord:
        print("Acorde detectado:", detected_chord)
    else:
        print("Acorde não reconhecido. Picos detectados:", detected_peaks)

    # Plot do sinal no domínio do tempo
    t = np.linspace(0, duration, N, endpoint=False)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(t[:1000], recording[:1000])
    axs[0].set_title("Sinal Gravado - Domínio do Tempo")
    axs[0].set_xlabel("Tempo (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Plot da FFT com picos
    axs[1].plot(pos_freq, pos_mag, label="FFT")
    axs[1].plot(pos_freq[peaks], pos_mag[peaks], "x", label="Picos")
    axs[1].set_title("Transformada de Fourier do Sinal Captado")
    axs[1].set_xlabel("Frequência (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlim(450, 1500)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
