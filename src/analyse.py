from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from scipy import interpolate, stats
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
    """
    signal: np.ndarray
    fs: float
    time_start: float
    
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
    Klasa odpowiedzialna za wczytywanie sygnałów z plików CSV
    
    Attributes:
        data_path (Path): Ścieżka do katalogu z danymi
    """
    def __init__(self, data_dir: str = "data"):
        self.data_path = Path().resolve() / data_dir
        # self.data_path = Path().resolve() / data_dir
    
    @staticmethod
    def _replace_nans(signal: np.ndarray) -> np.ndarray:
        """
        Zastępuje wartości NaN poprzednią wartością
        
        Args:
            signal: Tablica z wartościami sygnału
            
        Returns:
            Tablica z zastąpionymi wartościami NaN
        """
        signal = np.array(signal)
        mask = np.isnan(signal)
        
        # Jeśli nie ma NaN, zwróć oryginalny sygnał
        if not np.any(mask):
            return signal
            
        # Znajdź pierwszą nie-NaN wartość dla inicjalizacji
        first_valid_idx = np.where(~mask)[0]
        if len(first_valid_idx) == 0:
            raise ValueError("Sygnał zawiera same wartości NaN!")
        
        first_valid_value = signal[first_valid_idx[0]]
        
        # Zastąp NaN poprzednią wartością
        cleaned_signal = signal.copy()
        last_valid_value = first_valid_value
        
        for i in range(len(signal)):
            if np.isnan(signal[i]):
                cleaned_signal[i] = last_valid_value
            else:
                last_valid_value = signal[i]
                
        return cleaned_signal
    
    @staticmethod
    def _convert_datetime_to_time(datetime: np.ndarray, multi_day: bool = False) -> Tuple[np.ndarray, float]:
        """
        Konwertuje wartości datetime na wektor czasu i częstotliwość próbkowania
        
        Args:
            datetime: Tablica wartości datetime
            multi_day: Czy dane obejmują wiele dni
            
        Returns:
            Tuple zawierający wektor czasu i częstotliwość próbkowania
        """
        if not multi_day:
            t0 = (datetime[0] - np.floor(datetime[0])) * 24 * 3600
            t_hat = np.squeeze((datetime - np.floor(datetime)) * 24 * 3600 - t0)
            fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
        else:
            n_datetime = datetime - datetime[0]
            n_datetime_days = np.floor(n_datetime)
            c_datetime = n_datetime - n_datetime_days
            c_datetime_seconds = c_datetime * 24 * 3600

            t_hat = []
            for idx in range(0, len(datetime)):
                c_t = n_datetime_days[idx] * 24 * 3600 + c_datetime_seconds[idx]
                t_hat.append(c_t)
            t_hat = np.asarray(t_hat)
            fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
        return t_hat, fs_hat

    def load_signals(self, filename: str, multi_day: bool = False) -> Tuple[Optional[SignalData], Optional[SignalData]]:
        """
        Wczytuje sygnały ICP i ABP z pliku CSV
        
        Args:
            filename: Nazwa pliku CSV
            multi_day: Czy dane obejmują wiele dni
            
        Returns:
            Tuple zawierająca obiekty SignalData dla ICP i ABP lub None w przypadku błędu
        """
        try:
            file_path = self.data_path / filename
            # Wczytanie danych z CSV
            data = pd.read_csv(file_path)
            
            # Konwersja kolumny DateTime na wartości numeryczne
            datetime_values = pd.to_numeric(data['DateTime'])
            
            # Konwersja czasu i obliczenie częstotliwości próbkowania
            time_vector, fs = self._convert_datetime_to_time(datetime_values.values, multi_day)
            
            # Wczytanie i oczyszczenie sygnałów z NaN
            icp_values = self._replace_nans(data['icp[mmHg]'].values)
            abp_values = self._replace_nans(data['abp[mmHg]'].values)
            
            # Utworzenie obiektów SignalData dla obu sygnałów
            icp_signal = SignalData(
                signal=icp_values,
                fs=fs,
                time_start=time_vector[0]
            )
            
            abp_signal = SignalData(
                signal=abp_values,
                fs=fs,
                time_start=time_vector[0]
            )
            
            return icp_signal, abp_signal
            
        except FileNotFoundError:
            print(f"Nie znaleziono pliku: {file_path}")
            return None, None
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku {filename}: {str(e)}")
            return None, None

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
    
        t_icp = self.icp.time_vector + self.icp.time_start
        t_abp = self.abp.time_vector + self.abp.time_start
        
        if time_window:
            mask_icp = (t_icp >= time_window[0]) & (t_icp <= time_window[1])
            mask_abp = (t_abp >= time_window[0]) & (t_abp <= time_window[1])
            t_icp = t_icp[mask_icp]
            t_abp = t_abp[mask_abp]
            icp_signal = self.icp.signal[mask_icp]
            abp_signal = self.abp.signal[mask_abp]
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
                       db_scale: bool = True,
                       frequency_bands: Optional[dict] = None):
        """
        Porównuje widma FFT sygnałów ICP i ABP z opcjonalnym zaznaczeniem pasm częstotliwości
        
        Args:
            time_window: Opcjonalna krotka (start, koniec) w sekundach dla analizy
            freq_range: Opcjonalna krotka (min_freq, max_freq) do wyświetlenia
            db_scale: Czy używać skali decybelowej (logarytmicznej)
            frequency_bands: Słownik z pasmami częstotliwości w formacie:
                {'nazwa_pasma': (częstotliwość_min, częstotliwość_max)}
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
            mag_icp = 20 * np.log10(mag_icp + 1e-10)
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
        ax2.plot(freq_abp, mag_abp, 'r-', label='ABP')

        # Dodanie zaznaczenia pasm częstotliwości jeśli podano
        if frequency_bands:
            colors = ['lightgreen', 'lightblue', 'lightpink', 'lightyellow', 'lightgray']
            for (band_name, (f_min, f_max)), color in zip(frequency_bands.items(), colors):
                # Zaznaczenie na górnym wykresie (ICP)
                ax1.axvspan(f_min, f_max, alpha=0.3, color=color, label=band_name)
                # Zaznaczenie na dolnym wykresie (ABP)
                ax2.axvspan(f_min, f_max, alpha=0.3, color=color, label=band_name)

        ax1.set_ylabel('Amplituda ICP ' + ('[dB]' if db_scale else ''))
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel('Częstotliwość [Hz]')
        ax2.set_ylabel('Amplituda ABP ' + ('[dB]' if db_scale else ''))
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def analyze_frequency_bands(self, bands: dict = None):
        """
        Analizuje energię w określonych pasmach częstotliwości z dodatkową diagnostyką
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
            if np.sum(mask_icp) > 0:
                band_power = np.sum(mag_icp[mask_icp]**2)
                results['ICP'][band_name] = band_power
            else:
                results['ICP'][band_name] = 0

            # Dla ABP
            mask_abp = (freq_abp >= f_min) & (freq_abp <= f_max)
            if np.sum(mask_abp) > 0:
                band_power = np.sum(mag_abp[mask_abp]**2)
                results['ABP'][band_name] = band_power
            else:
                results['ABP'][band_name] = 0

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
        
        # Sprawdzenie czy częstotliwości próbkowania są takie same
        if self.icp.fs != self.abp.fs:
            raise ValueError("Sygnały muszą mieć tę samą częstotliwość próbkowania")
            
        # Obliczenie koherencji
        freq, coh = signal.coherence(self.icp.signal, self.abp.signal,
                                   fs=self.icp.fs,
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   window=window)
        
        # Obliczenie fazy
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
        Wizualizuje analizę koherencji między sygnałami z istotności statystyczną per segment
        
        Args:
            freq_range (tuple, optional): Zakres częstotliwości (min, max) w Hz
            nperseg (int): Długość segmentu do analizy
            significance_level (float): Poziom istotności statystycznej (0-1)
        """
        # Obliczenie koherencji używając scipy
        freq, coherence, phase = self.calculate_coherence(nperseg=nperseg)
        
        # Obliczenie efektywnej liczby segmentów (uwzględniając nakładanie się)
        noverlap = nperseg // 2
        n_segments = int(np.floor((len(self.icp.signal) - noverlap) / (nperseg - noverlap)))
        # Korekta na nakładanie się segmentów
        n_eff = n_segments * (1 - noverlap/nperseg)  # efektywna liczba niezależnych segmentów
        
        # Obliczenie p-value dla każdej częstotliwości
        # Używamy skorygowanej liczby segmentów i mniej agresywnej transformacji
        z_transform = np.sqrt(n_eff) * np.arctanh(coherence)  # usunięte mnożenie przez 2 i sqrt z coherence
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_transform)))
        
        # Przygotowanie wykresu
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Ograniczenie zakresu częstotliwości jeśli podano
        if freq_range:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[mask]
            coherence = coherence[mask]
            phase = phase[mask]
            p_values = p_values[mask]

        # Wykres koherencji
        ax1.plot(freq, coherence, 'b-', label='Koherencja')
        ax1.set_ylabel('Koherencja')
        ax1.grid(True)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Wykres fazy
        ax2.plot(freq, phase, 'g-', label='Faza')
        ax2.set_ylabel('Faza [rad]')
        ax2.grid(True)
        ax2.legend()
        ax2.set_ylim(-np.pi, np.pi)

        # Wykres p-value
        ax3.plot(freq, p_values, 'r-', label='p-value')
        critical_value = 1 - significance_level  # = 0.05 dla significance_level = 0.95
        ax3.axhline(y=critical_value, color='k', linestyle='--', 
                    label=f'Poziom krytyczny = {critical_value:.3f}')
        
        # Zaznaczenie obszarów istotnych statystycznie
        # Zwiększamy alpha dla lepszej widoczności i upewniamy się, że wypełnienie jest pod krzywą p-value
        significant_mask = p_values < critical_value
        ax3.fill_between(freq, 0, 1,  # wypełniamy całą wysokość wykresu
                        where=significant_mask,
                        color='g', alpha=0.2,
                        label='Obszar istotny statystycznie')
        
        ax3.set_xlabel('Częstotliwość [Hz]')
        ax3.set_ylabel('p-value')
        ax3.grid(True)
        ax3.legend()
        ax3.set_ylim(0, 1)

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
            nperseg=4096,  # Zwiększona długość okna
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
    icp_signal, abp_signal = loader.load_signals("PAC7_r3_SHORT.csv", True)
    
    # Jeżeli oba sygnały istnieją
    if icp_signal and abp_signal:
        # Utworzenie analizatora
        analyzer = SignalAnalyzer(icp_signal, abp_signal)
        
        # Wyświetlenie podstawowych informacji
        print(f"Częstotliwość próbkowania\nICP: {icp_signal.fs} Hz \nABP: {abp_signal.fs} Hz \n")
        print(f"Długość sygnału\nICP: {icp_signal.duration:.2f} s \nABP: {abp_signal.duration} s \n")
        print(f"Czas rozpoczęcia\nICP: {icp_signal.time_start:.2f} s \nABP: {abp_signal.time_start} s \n")

        # Wyświetlenie pierwszych 10 sekund sygnałów
        analyzer.plot_signals(time_window=(0, 10))

        # Analiza FFT 
        analyzer.plot_fft_comparison(
            # time_window=(0, 60),  # ograniczenie analizy FFT do pierwszych 60s
            freq_range=(0, 0.5),  # Pokazuje częstotliwości do 0.5 Hz
            db_scale=True,
            frequency_bands={
                'VLF': (0.02, 0.07),
                'LF': (0.07, 0.2),
                'HF': (0.2, 0.5)
            }
        )

        analyzer.plot_fft_comparison(
            # time_window=(0, 60),  # ograniczenie analizy FFT do pierwszych 60s
            freq_range=(0, 0.5),  # Pokazuje częstotliwości do 0.5 Hz
            db_scale=False,
            frequency_bands={
                'VLF': (0.02, 0.07),
                'LF': (0.07, 0.2),
                'HF': (0.2, 0.5)
            }
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