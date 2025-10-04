/**
 * Signal Visualization Component using Chart.js
 */
import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { motion } from 'framer-motion';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const SignalVisualization = ({ ppgData, ecgData, predictedABP = null }) => {
  // Chart configuration
  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        }
      },
      title: {
        display: true,
        text: 'Physiological Signals',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          title: (context) => `Sample ${context[0].label}`,
          label: (context) => {
            const label = context.dataset.label || '';
            return `${label}: ${context.parsed.y.toFixed(3)}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Sample Number',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        // Show fewer ticks to avoid crowding
        ticks: {
          maxTicksLimit: 10
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Amplitude',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    elements: {
      line: {
        tension: 0.1,
        borderWidth: 2
      },
      point: {
        radius: 0,
        hoverRadius: 4
      }
    },
    animation: {
      duration: 1000,
      easing: 'easeOutQuart'
    }
  }), []);

  // Prepare chart data
  const chartData = useMemo(() => {
    const labels = Array.from({ length: 1000 }, (_, i) => i + 1);
    
    const datasets = [
      {
        label: 'PPG Signal',
        data: ppgData,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: false,
        tension: 0.1
      },
      {
        label: 'ECG Signal',
        data: ecgData,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: false,
        tension: 0.1
      }
    ];

    // Add predicted ABP if available
    if (predictedABP && Array.isArray(predictedABP)) {
      datasets.push({
        label: 'Predicted ABP',
        data: predictedABP,
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: false,
        tension: 0.1,
        borderDash: [5, 5]
      });
    }

    return {
      labels,
      datasets
    };
  }, [ppgData, ecgData, predictedABP]);

  // Signal statistics
  const signalStats = useMemo(() => {
    const calculateStats = (signal) => {
      const values = signal.filter(v => !isNaN(v) && isFinite(v));
      if (values.length === 0) return { mean: 0, std: 0, min: 0, max: 0 };
      
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);
      const min = Math.min(...values);
      const max = Math.max(...values);
      
      return { mean, std, min, max };
    };

    return {
      ppg: calculateStats(ppgData),
      ecg: calculateStats(ecgData)
    };
  }, [ppgData, ecgData]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full"
    >
      {/* Chart */}
      <div className="h-96 mb-6">
        <Line options={chartOptions} data={chartData} />
      </div>

      {/* Signal Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* PPG Statistics */}
        <div className="bg-red-50 rounded-lg p-4 border border-red-200">
          <h3 className="font-semibold text-red-800 mb-2 flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
            PPG Signal Statistics
          </h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="text-gray-700">
              <span className="font-medium">Mean:</span> {signalStats.ppg.mean.toFixed(3)}
            </div>
            <div className="text-gray-700">
              <span className="font-medium">Std:</span> {signalStats.ppg.std.toFixed(3)}
            </div>
            <div className="text-gray-700">
              <span className="font-medium">Min:</span> {signalStats.ppg.min.toFixed(3)}
            </div>
            <div className="text-gray-700">
              <span className="font-medium">Max:</span> {signalStats.ppg.max.toFixed(3)}
            </div>
          </div>
        </div>

        {/* ECG Statistics */}
        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
          <h3 className="font-semibold text-blue-800 mb-2 flex items-center">
            <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
            ECG Signal Statistics
          </h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="text-gray-700">
              <span className="font-medium">Mean:</span> {signalStats.ecg.mean.toFixed(3)}
            </div>
            <div className="text-gray-700">
              <span className="font-medium">Std:</span> {signalStats.ecg.std.toFixed(3)}
            </div>
            <div className="text-gray-700">
              <span className="font-medium">Min:</span> {signalStats.ecg.min.toFixed(3)}
            </div>
            <div className="text-gray-700">
              <span className="font-medium">Max:</span> {signalStats.ecg.max.toFixed(3)}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default SignalVisualization;
