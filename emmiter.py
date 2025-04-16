import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

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
    print("Escolha um acorde para emissão:")
    for idx, chord in enumerate(chords.keys()):
        print(f"{idx+1}: {chord}")
    try:
        opcao = int(input("Digite o número do acorde desejado: "))
        if opcao < 1 or opcao > len(chords):
            print("Opção inválida!")
            return
    except ValueError:
        print("Entrada inválida!")
        return
    
    chord_name = list(chords.keys())[opcao - 1]
    freq_list = chords[chord_name]
    print(f"Acorde escolhido: {chord_name} - Frequências: {freq_list}")

    fs = 44100
    duration = 3
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Geração do sinal com amplitudes iguais
    signal = np.zeros_like(t)
    for f in freq_list:
        signal += np.sin(2 * np.pi * f * t)
    signal = signal / len(freq_list)  # Normaliza pela quantidade de senoides
    signal = signal / np.max(np.abs(signal)) * 0.9  # Normaliza para 90% da amplitude máxima

    print("Reproduzindo áudio...")
    sd.play(signal, fs)
    sd.wait()

    # Plot do sinal no domínio do tempo
    plt.figure(figsize=(10, 4))
    plt.plot(t[:1000], signal[:1000])
    plt.title(f"Sinal no Domínio do Tempo - {chord_name}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Cálculo e plot da FFT
    N = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(N, d=1/fs)
    idx_positive = np.where(fft_freq >= 0)
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq[idx_positive], np.abs(fft_signal[idx_positive]))
    plt.title(f"Transformada de Fourier do Sinal Emitido - {chord_name}")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1500)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
