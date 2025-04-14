import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import find_peaks

# Mesmo dicionário de acordes para comparar os picos detectados
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
    fs = 44100         # Taxa de amostragem (Hz)
    duration = 3       # Duração da gravação (segundos)

    print("Captando o sinal de áudio via microfone...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Aguarda o término da gravação
    recording = recording.flatten()  # Converte para vetor unidimensional

    # Ajuste de ganho se o volume estiver baixo
    max_amp = np.max(np.abs(recording))
    print("Amplitude máxima capturada:", max_amp)
    if max_amp < 0.5:
        print("Volume baixo: ajustando o ganho do sinal captado.")
        recording *= 2

    # Cálculo da FFT do sinal captado
    N = len(recording)
    fft_rec = np.fft.fft(recording)
    fft_freq = np.fft.fftfreq(N, d=1/fs)
    magnitude = np.abs(fft_rec)
    
    # Seleciona somente as frequências positivas
    idx_positive = np.where(fft_freq >= 0)
    pos_freq = fft_freq[idx_positive]
    pos_mag = magnitude[idx_positive]
    
    # Identificação dos picos na FFT
    peaks, properties = find_peaks(pos_mag, height=np.max(pos_mag) * 0.1 ,distance=fs//551)
    if len(peaks) < 5:
        peaks, properties = find_peaks(pos_mag, height=np.max(pos_mag) * 0.05, distance=fs//551)
    print("Picos encontrados (Hz):", np.round(pos_freq[peaks], 2))
    
    # Tentativa de identificar o acorde com uma tolerância de ±5 Hz
    tolerance = 5
    detected_chord = None
    for chord, freqs in chords.items():
        count = 0
        for f in freqs:
            if np.any(np.abs(pos_freq[peaks] - f) < tolerance):
                count += 1
        # Se pelo menos duas frequências coincidirem, consideramos que é o acorde detectado
        if count >= 2:
            detected_chord = chord
            break

    if detected_chord:
        print("Acorde detectado:", detected_chord)
    else:
        print("Acorde não reconhecido.")

    # Plot do sinal captado no domínio do tempo
    t = np.linspace(0, duration, N, endpoint=False)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 linhas, 1 coluna

    # Gráfico do sinal no domínio do tempo
    axs[0].plot(t, recording)
    axs[0].set_title("Sinal Gravado - Domínio do Tempo")
    axs[0].set_xlabel("Tempo (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Gráfico da FFT com os picos identificados
    axs[1].plot(pos_freq, pos_mag, label="FFT")
    axs[1].plot(pos_freq[peaks], pos_mag[peaks], "x", label="Picos")
    axs[1].set_title("Transformada de Fourier do Sinal Captado")
    axs[1].set_xlabel("Frequência (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.show()

if __name__ == '__main__':
    main()
