from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@dataclass
class SignalData:
    """
    Klasa przechowująca dane pojedynczego sygnału biomedycznego
    
    Attributes:
        signal (np.ndarray): Tablica z wartościami sygnału
        fs (float): Częstotliwość próbkowania w Hz
        time_start (float): Czas rozpoczęcia rejestracji w sekundach
        any_error_flag (bool): Flaga wskazująca na wystąpienie błędów
        percent_invalid (float): Procent nieprawidłowych próbek
    """
    signal: np.ndarray
    fs: float
    time_start: float
    any_error_flag: bool
    percent_invalid: float
    
    @property
    def duration(self) -> float:
        """Zwraca długość sygnału w sekundach"""
        return len(self.signal) / self.fs
    
    @property
    def time_vector(self) -> np.ndarray:
        """Generuje wektor czasu dla sygnału"""
        return np.arange(len(self.signal)) / self.fs

class SignalLoader:
    """
    Klasa odpowiedzialna za wczytywanie sygnałów z plików PKL
    
    Attributes:
        data_path (Path): Ścieżka do katalogu z danymi
    """
    def __init__(self, data_dir: str = "data"):
        self.data_path = Path().resolve().parent / data_dir
        
    def load_signal(self, filename: str, signal_index: int = 0) -> Optional[SignalData]:
        """
        Wczytuje sygnał z pliku PKL
        
        Args:
            filename: Nazwa pliku PKL
            signal_index: Indeks sygnału do wczytania (domyślnie 0)
            
        Returns:
            SignalData lub None w przypadku błędu
        """
        try:
            file_path = self.data_path / filename
            data = pd.read_pickle(file_path)
            
            return SignalData(
                signal=data[signal_index]["signal"],
                fs=data[signal_index]["fs"],
                time_start=data[signal_index]["time_start"],
                any_error_flag=data[signal_index]["any_error_flag"],
                percent_invalid=data[signal_index]["percent_invalid"]
            )
        except FileNotFoundError:
            print(f"Nie znaleziono pliku: {file_path}")
            return None
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku {filename}: {str(e)}")
            return None

class SignalAnalyzer:
    """
    Klasa do analizy pary sygnałów ICP (ciśnienie śródczaszkowe) i ABP (ciśnienie tętnicze)
    
    Attributes:
        icp (SignalData): Obiekt zawierający dane sygnału ICP
        abp (SignalData): Obiekt zawierający dane sygnału ABP
    """

    def __init__(self, icp_signal: SignalData, abp_signal: SignalData):
        self.icp = icp_signal
        self.abp = abp_signal
        
    def plot_signals(self, time_window: Optional[tuple] = None):
        """
        Wyświetla wykresy obu sygnałów w czasie
        
        Args:
            time_window (tuple, optional): Zakres czasu do wyświetlenia (start, koniec) w sekundach
            
        Notes:
            - Tworzy dwa podwykresy: ICP na górze, ABP na dole
            - Wyświetla siatkę i etykiety osi
            - Jeśli time_window nie podano, wyświetla cały sygnał
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        t_icp = self.icp.time_vector
        t_abp = self.abp.time_vector
        
        if time_window:
            start_idx = int(time_window[0] * self.icp.fs)
            end_idx = int(time_window[1] * self.icp.fs)
            t_icp = t_icp[start_idx:end_idx]
            t_abp = t_abp[start_idx:end_idx]
            icp_signal = self.icp.signal[start_idx:end_idx]
            abp_signal = self.abp.signal[start_idx:end_idx]
        else:
            icp_signal = self.icp.signal
            abp_signal = self.abp.signal
        
        ax1.plot(t_icp, icp_signal)
        ax1.set_ylabel('ICP [mmHg]')
        ax1.grid(True)
        
        ax2.plot(t_abp, abp_signal)
        ax2.set_ylabel('ABP [mmHg]')
        ax2.set_xlabel('Czas [s]')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_fft(self, signal: np.ndarray, fs: float):
        """
        Oblicza transformatę Fouriera dla sygnału
        
        Args:
            signal (np.ndarray): Sygnał wejściowy
            fs (float): Częstotliwość próbkowania w Hz
            
        Returns:
            tuple: (frequencies, magnitudes)
                - frequencies (np.ndarray): Wektor częstotliwości w Hz
                - magnitudes (np.ndarray): Znormalizowane amplitudy widma
        
        Notes:
            - Używa FFT z biblioteki numpy
            - Normalizuje amplitudy przez długość sygnału
            - Zwraca tylko dodatnie częstotliwości (do Nyquista)
        """
        n = len(signal)
        fft_result = np.fft.fft(signal)
        magnitudes = np.abs(fft_result)[:n//2] * 2 / n  # Normalizacja amplitudy
        frequencies = np.fft.fftfreq(n, 1/fs)[:n//2]
        
        return frequencies, magnitudes

    def plot_fft_comparison(self, time_window: Optional[tuple] = None, 
                          freq_range: Optional[tuple] = None,
                          db_scale: bool = True):
        """
        Porównuje widma FFT sygnałów ICP i ABP
        
        Args:
            time_window: Opcjonalna krotka (start, koniec) w sekundach dla analizy
            freq_range: Opcjonalna krotka (min_freq, max_freq) do wyświetlenia
            db_scale: Czy używać skali decybelowej (logarytmicznej)
        """
        # Przygotowanie danych
        if time_window:
            start_idx = int(time_window[0] * self.icp.fs)
            end_idx = int(time_window[1] * self.icp.fs)
            icp_signal = self.icp.signal[start_idx:end_idx]
            abp_signal = self.abp.signal[start_idx:end_idx]
        else:
            icp_signal = self.icp.signal
            abp_signal = self.abp.signal

        # Obliczenie FFT dla obu sygnałów
        freq_icp, mag_icp = self.calculate_fft(icp_signal, self.icp.fs)
        freq_abp, mag_abp = self.calculate_fft(abp_signal, self.abp.fs)

        # Konwersja do skali dB jeśli wymagane
        if db_scale:
            mag_icp = 20 * np.log10(mag_icp + 1e-10)  # Dodajemy małą wartość aby uniknąć log(0)
            mag_abp = 20 * np.log10(mag_abp + 1e-10)

        # Przygotowanie wykresu
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Ograniczenie zakresu częstotliwości jeśli podano
        if freq_range:
            freq_mask_icp = (freq_icp >= freq_range[0]) & (freq_icp <= freq_range[1])
            freq_mask_abp = (freq_abp >= freq_range[0]) & (freq_abp <= freq_range[1])
            freq_icp = freq_icp[freq_mask_icp]
            freq_abp = freq_abp[freq_mask_abp]
            mag_icp = mag_icp[freq_mask_icp]
            mag_abp = mag_abp[freq_mask_abp]

        # Wykreślenie widm
        ax1.plot(freq_icp, mag_icp, 'b-', label='ICP')
        ax1.set_ylabel('Amplituda ICP ' + ('[dB]' if db_scale else ''))
        ax1.grid(True)
        ax1.legend()

        ax2.plot(freq_abp, mag_abp, 'r-', label='ABP')
        ax2.set_xlabel('Częstotliwość [Hz]')
        ax2.set_ylabel('Amplituda ABP ' + ('[dB]' if db_scale else ''))
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def analyze_frequency_bands(self, bands: dict = None):
        """
        Analizuje energię w określonych pasmach częstotliwości
        
        Args:
            bands: Słownik z definicjami pasm częstotliwości, np.:
                  {'VLF': (0.02, 0.07), 'LF': (0.07, 0.2), 'HF': (0.2, 0.5)}
        Returns:
            dict: Słownik z energią w każdym paśmie dla obu sygnałów
        """
        if bands is None:
            bands = {
                'VLF': (0.02, 0.07),
                'LF': (0.07, 0.2),
                'HF': (0.2, 0.5)
            }

        # Obliczenie FFT dla obu sygnałów
        freq_icp, mag_icp = self.calculate_fft(self.icp.signal, self.icp.fs)
        freq_abp, mag_abp = self.calculate_fft(self.abp.signal, self.abp.fs)

        results = {'ICP': {}, 'ABP': {}}

        # Analiza każdego pasma
        for band_name, (f_min, f_max) in bands.items():
            # Dla ICP
            mask_icp = (freq_icp >= f_min) & (freq_icp <= f_max)
            results['ICP'][band_name] = np.sum(mag_icp[mask_icp]**2)

            # Dla ABP
            mask_abp = (freq_abp >= f_min) & (freq_abp <= f_max)
            results['ABP'][band_name] = np.sum(mag_abp[mask_abp]**2)

        return results
    
    def calculate_coherence(self, nperseg: int = 256, noverlap: int = None, window: str = 'hann'):
        """
        Oblicza koherencję między sygnałami ICP i ABP
        
        Args:
            nperseg (int): Długość segmentu do analizy (w próbkach)
            noverlap (int, optional): Liczba próbek nakładania się segmentów
            window (str): Typ okna do analizy (np. 'hann', 'hamming')
            
        Returns:
            tuple: (frequencies, coherence, phase)
                - frequencies (np.ndarray): Wektor częstotliwości
                - coherence (np.ndarray): Wartości koherencji (0-1)
                - phase (np.ndarray): Faza w radianach (-π do π)
                
        Notes:
            - Wykorzystuje scipy.signal.coherence do obliczenia koherencji
            - Faza obliczana jest za pomocą scipy.signal.csd
        """
        from scipy import signal
        
        # Użycie wbudowanej funkcji scipy do obliczenia koherencji
        freq, coh = signal.coherence(self.icp.signal, self.abp.signal, 
                                fs=self.icp.fs,
                                nperseg=nperseg,
                                noverlap=noverlap,
                                window=window)
        
        # Obliczenie fazy używając csd
        _, phase = signal.csd(self.icp.signal, self.abp.signal, 
                            fs=self.icp.fs,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            window=window)
        phase = np.angle(phase)
        
        return freq, coh, phase


    def plot_coherence_analysis(self, freq_range: Optional[tuple] = None, 
                              nperseg: int = 256, significance_level: float = 0.95):
        """
        Wizualizuje analizę koherencji między sygnałami
        
        Args:
            freq_range (tuple, optional): Zakres częstotliwości (min, max) w Hz
            nperseg (int): Długość segmentu do analizy
            significance_level (float): Poziom istotności statystycznej (0-1)
            
        Notes:
            - Górny wykres pokazuje koherencję z poziomem istotności
            - Dolny wykres pokazuje fazę między sygnałami
            - Automatycznie oblicza próg istotności statystycznej
        """
        # Obliczenie koherencji używając scipy
        freq, coherence, phase = self.calculate_coherence(nperseg=nperseg)
        
        # Obliczenie poziomu istotności statystycznej
        dof = 2 * len(self.icp.signal) / nperseg
        significance_threshold = 1 - (1 - significance_level)**(1/(dof-1))

        # Przygotowanie wykresu
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Ograniczenie zakresu częstotliwości jeśli podano
        if freq_range:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[mask]
            coherence = coherence[mask]
            phase = phase[mask]

        # Wykres koherencji
        ax1.plot(freq, coherence, 'b-', label='Koherencja')
        ax1.axhline(y=significance_threshold, color='r', linestyle='--', 
                   label=f'Poziom istotności ({significance_level:.2f})')
        ax1.set_ylabel('Koherencja')
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Wykres fazy
        ax2.plot(freq, phase, 'g-', label='Faza')
        ax2.set_xlabel('Częstotliwość [Hz]')
        ax2.set_ylabel('Faza [rad]')
        ax2.grid(True)
        ax2.legend()
        ax2.set_ylim(-np.pi, np.pi)

        plt.tight_layout()
        plt.show()
    
    def analyze_specific_bands(self):
        """
        Analizuje koherencję w specyficznych pasmach częstotliwości
        
        Returns:
            dict: Słownik z wynikami dla każdego pasma częstotliwości:
                - mean_coherence: Średnia koherencja w paśmie
                - max_coherence: Maksymalna koherencja
                - mean_phase: Średnia faza
                - phase_at_max_coh: Faza przy maksymalnej koherencji
                - freq_range: Zakres częstotliwości pasma
                
        Notes:
            - Analizuje pasma: VLF (0.02-0.07 Hz), LF (0.07-0.2 Hz), HF (0.2-0.5 Hz)
            - Obsługuje przypadki braku danych w pasmach
            - Używa zwiększonej rozdzielczości częstotliwościowej (nperseg=2048)
        """
        # Obliczenie koherencji z większą rozdzielczością częstotliwościową
        freq, coh, phase = self.calculate_coherence(nperseg=2048)  # Zwiększona długość okna
        
        # Definicja pasm częstotliwości
        bands = {
            'VLF': (0.02, 0.07),
            'LF': (0.07, 0.2),
            'HF': (0.2, 0.5)
        }
        
        results = {}
        for band_name, (f_min, f_max) in bands.items():
            # Znalezienie indeksów dla danego pasma
            band_mask = (freq >= f_min) & (freq <= f_max)
            
            # Sprawdzenie czy mamy jakieś punkty w tym paśmie
            if not np.any(band_mask):
                results[band_name] = {
                    'mean_coherence': np.nan,
                    'max_coherence': np.nan,
                    'mean_phase': np.nan,
                    'phase_at_max_coh': np.nan,
                    'freq_range': (f_min, f_max)
                }
                continue
                
            band_coh = coh[band_mask]
            band_phase = phase[band_mask]
            band_freq = freq[band_mask]
            
            # Znalezienie indeksu maksymalnej koherencji
            if len(band_coh) > 0:
                max_coh_idx = np.argmax(band_coh)
                max_coh = band_coh[max_coh_idx]
                phase_at_max = band_phase[max_coh_idx]
            else:
                max_coh = np.nan
                phase_at_max = np.nan
                
            results[band_name] = {
                'mean_coherence': np.mean(band_coh),
                'max_coherence': max_coh,
                'mean_phase': np.mean(band_phase),
                'phase_at_max_coh': phase_at_max,
                'freq_range': (f_min, f_max)
            }
        
        return results
    
    def check_signal_quality(self):
        """
        Sprawdza jakość sygnałów przed analizą
        """
        min_length = 60 * self.icp.fs  # minimum 60 sekund danych
        
        if len(self.icp.signal) < min_length or len(self.abp.signal) < min_length:
            print("UWAGA: Sygnały są zbyt krótkie dla wiarygodnej analizy")
            return False
            
        if self.icp.percent_invalid > 10 or self.abp.percent_invalid > 10:
            print("UWAGA: Wysoki procent nieprawidłowych próbek może wpływać na wyniki")
            return False
            
        return True

    def analyze_icp_abp_coherence(self):
        """
        Pełna analiza koherencji dla sygnałów ICP-ABP
        """
        print("\n=== ANALIZA KOHERENCJI ICP-ABP ===")
        if not self.check_signal_quality():
            print("Analiza może być niewiarygodna ze względu na jakość sygnału")
        
        # 1. Podstawowa analiza koherencji
        self.plot_coherence_analysis(
            freq_range=(0, 0.5),
            nperseg=2048,  # Zwiększona długość okna
            significance_level=0.95
        )
        
        # 2. Szczegółowa analiza w pasmach
        band_results = self.analyze_specific_bands()
        
        # 3. Wyświetlenie wyników
        print("\nWyniki analizy koherencji w pasmach częstotliwości:")
        for band, results in band_results.items():
            print(f"\n{band} ({results['freq_range'][0]:.3f}-{results['freq_range'][1]:.3f} Hz):")
            if np.isnan(results['mean_coherence']):
                print("  Brak wystarczających danych w tym paśmie")
            else:
                print(f"  Średnia koherencja: {results['mean_coherence']:.3f}")
                print(f"  Maksymalna koherencja: {results['max_coherence']:.3f}")
                print(f"  Średnia faza: {results['mean_phase']:.3f} rad")
                print(f"  Faza przy max koherencji: {results['phase_at_max_coh']:.3f} rad")

def main():
    """
    Główna funkcja programu wykonująca pełną analizę sygnałów ICP i ABP
    
    Wykonuje następujące kroki:
    1. Wczytuje sygnały z plików PKL
    2. Wyświetla podstawowe informacje o sygnałach
    3. Pokazuje pierwsze 10 sekund sygnałów
    4. Wykonuje analizę FFT dla pierwszych 60 sekund
    5. Analizuje energię w pasmach częstotliwości
    6. Przeprowadza pełną analizę koherencji
    
    Notes:
        - Wymaga plików z sygnałami w katalogu 'data'
        - Wyświetla wykresy i wyniki w konsoli
    """
    loader = SignalLoader()
    
    # Wczytanie sygnałów
    icp_signal = loader.load_signal("2aHc688_ICP.pkl", 1)
    abp_signal = loader.load_signal("2aHc688_ABP.pkl", 1)
    
    if icp_signal and abp_signal:
        # Utworzenie analizatora
        analyzer = SignalAnalyzer(icp_signal, abp_signal)
        
        # Wyświetlenie podstawowych informacji
        print(f"Częstotliwość próbkowania\nICP: {icp_signal.fs} Hz \nABP: {abp_signal.fs} Hz \n")
        print(f"Długość sygnału\nICP: {icp_signal.duration:.2f} s \nABP: {abp_signal.duration} s \n")
        print(f"Czas rozpoczęcia\nICP: {icp_signal.time_start:.2f} s \nABP: {abp_signal.time_start} s \n")
        print(f"Procent nieprawidłowych próbek\nICP: {icp_signal.percent_invalid}% \nABP: {abp_signal.percent_invalid}%")

        # Wyświetlenie pierwszych 10 sekund sygnałów
        analyzer.plot_signals(time_window=(0, 10))

        # Analiza FFT dla pierwszych 60 sekund sygnału
        analyzer.plot_fft_comparison(
            time_window=(0, 60),
            freq_range=(0, 2),  # Pokazuje częstotliwości do 2 Hz
            db_scale=True
        )
        
        # Analiza pasm częstotliwości
        bands_energy = analyzer.analyze_frequency_bands()
        
        # Wyświetlenie wyników
        print("\nEnergia w pasmach częstotliwości:")
        for signal_type, bands in bands_energy.items():
            print(f"\n{signal_type}:")
            for band_name, energy in bands.items():
                print(f"{band_name}: {energy:.2e}")
        
        # Pełna analiza koherencji dla sygnałów ICP-ABP
        analyzer.analyze_icp_abp_coherence()


if __name__ == "__main__":
    main()