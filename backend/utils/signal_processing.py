"""
Advanced signal processing utilities for PPG and ECG signals.
"""
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AdvancedSignalProcessor:
    """Advanced signal processor for quality assessment and preprocessing."""
    
    def __init__(self, sample_rate: int = 125):
        """
        Initialize the signal processor.
        
        Args:
            sample_rate: Sampling rate in Hz (default: 125)
        """
        self.sample_rate = sample_rate
        
    def assess_signal_quality(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of a signal.
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {}
        
        # Calculate SNR (Signal-to-Noise Ratio)
        quality_metrics['snr'] = self._calculate_snr(signal_data)
        
        # Calculate signal statistics
        quality_metrics['mean'] = float(np.mean(signal_data))
        quality_metrics['std'] = float(np.std(signal_data))
        quality_metrics['skewness'] = float(skew(signal_data))
        quality_metrics['kurtosis'] = float(kurtosis(signal_data))
        
        # Detect artifacts
        quality_metrics['artifact_ratio'] = self._detect_artifacts(signal_data)
        
        # Calculate overall quality score (0-1)
        quality_metrics['quality_score'] = self._calculate_quality_score(quality_metrics)
        
        return quality_metrics
    
    def _calculate_snr(self, signal_data: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        # Use moving average to estimate signal
        window_size = min(50, len(signal_data) // 10)
        signal_estimate = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
        noise = signal_data - signal_estimate
        
        signal_power = np.mean(signal_estimate ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return float(max(0, min(snr, 100)))
        return 100.0
    
    def _detect_artifacts(self, signal_data: np.ndarray) -> float:
        """Detect artifacts in signal."""
        z_scores = np.abs((signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10))
        artifact_mask = z_scores > 3.5
        artifact_ratio = float(np.sum(artifact_mask) / len(signal_data))
        return artifact_ratio
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics."""
        snr_score = min(metrics['snr'] / 40.0, 1.0)
        artifact_penalty = 1.0 - metrics['artifact_ratio']
        quality_score = 0.7 * snr_score + 0.3 * artifact_penalty
        return float(max(0.0, min(quality_score, 1.0)))
    
    def filter_signal(self, signal_data: np.ndarray, filter_type: str = 'bandpass',
                     lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
        """Apply filtering to signal."""
        nyquist = self.sample_rate / 2.0
        
        if filter_type == 'bandpass':
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
        elif filter_type == 'lowpass':
            high = highcut / nyquist
            b, a = signal.butter(4, high, btype='low')
        elif filter_type == 'highpass':
            low = lowcut / nyquist
            b, a = signal.butter(4, low, btype='high')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    
    def normalize_signal(self, signal_data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize signal."""
        if method == 'minmax':
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            if max_val - min_val > 0:
                normalized = (signal_data - min_val) / (max_val - min_val)
            else:
                normalized = signal_data
        elif method == 'zscore':
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            if std > 0:
                normalized = (signal_data - mean) / std
            else:
                normalized = signal_data - mean
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
