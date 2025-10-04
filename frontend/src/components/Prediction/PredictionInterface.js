/**
 * Main Prediction Interface Component
 */
import React, { useState, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'react-hot-toast';
import { 
  ChartBarIcon, 
  CloudUploadIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

// Components
import SignalInput from './SignalInput';
import SignalVisualization from './SignalVisualization';
import PredictionResults from './PredictionResults';
import QualityAssessment from './QualityAssessment';
import SampleDataLoader from './SampleDataLoader';

// Hooks and utilities
import { usePrediction } from '../../hooks/usePrediction';
import { validateSignalData } from '../../utils/signalValidation';
import { generateSampleData } from '../../utils/sampleDataGenerator';

const PredictionInterface = () => {
  const [ppgData, setPpgData] = useState(Array(1000).fill(0));
  const [ecgData, setEcgData] = useState(Array(1000).fill(0));
  const [includePPGuality, setIncludeQuality] = useState(true);
  const [isDataValid, setIsDataValid] = useState(false);
  
  const { prediction, isLoading, error, makePrediction, clearPrediction } = usePrediction();

  // Handle signal data changes
  const handlePPGChange = useCallback((newData) => {
    setPpgData(newData);
    validateData(newData, ecgData);
  }, [ecgData]);

  const handleECGChange = useCallback((newData) => {
    setEcgData(newData);
    validateData(ppgData, newData);
  }, [ppgData]);

  // Validate signal data
  const validateData = useCallback((ppgSignal, ecgSignal) => {
    try {
      const ppgValid = validateSignalData(ppgSignal, 'PPG');
      const ecgValid = validateSignalData(ecgSignal, 'ECG');
      setIsDataValid(ppgValid.isValid && ecgValid.isValid);
      
      if (!ppgValid.isValid) {
        toast.error(ppgValid.message);
      }
      if (!ecgValid.isValid) {
        toast.error(ecgValid.message);
      }
    } catch (error) {
      setIsDataValid(false);
      toast.error('Data validation failed');
    }
  }, []);

  // Handle prediction request
  const handlePredict = useCallback(async () => {
    if (!isDataValid) {
      toast.error('Please provide valid PPG and ECG data');
      return;
    }

    try {
      clearPrediction();
      
      const result = await makePrediction({
        ppg_signal: ppgData,
        ecg_signal: ecgData,
        include_quality_metrics: includePPGuality
      });

      if (result) {
        toast.success('Blood pressure prediction completed successfully!');
      }
    } catch (err) {
      toast.error(err.message || 'Prediction failed');
    }
  }, [ppgData, ecgData, includePPGuality, isDataValid, makePrediction, clearPrediction]);

  // Load sample data
  const handleLoadSampleData = useCallback((sampleType) => {
    try {
      const sampleData = generateSampleData(sampleType);
      setPpgData(sampleData.ppg);
      setEcgData(sampleData.ecg);
      validateData(sampleData.ppg, sampleData.ecg);
      toast.success(`Loaded ${sampleType} sample data`);
    } catch (error) {
      toast.error('Failed to load sample data');
    }
  }, [validateData]);

  // Clear all data
  const handleClearData = useCallback(() => {
    setPpgData(Array(1000).fill(0));
    setEcgData(Array(1000).fill(0));
    setIsDataValid(false);
    clearPrediction();
    toast.success('Data cleared');
  }, [clearPrediction]);

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Blood Pressure Prediction Tool
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Advanced AI-powered blood pressure prediction using PPG and ECG signals with 
          real-time signal quality assessment and comprehensive visualization.
        </p>
      </motion.div>

      {/* Main Interface Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Column - Input and Controls */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="space-y-6"
        >
          {/* Sample Data Loader */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <CloudUploadIcon className="w-6 h-6 mr-2 text-blue-600" />
              Sample Data & Input
            </h2>
            <SampleDataLoader onLoadSample={handleLoadSampleData} />
          </div>

          {/* Signal Input */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Signal Input
            </h2>
            <SignalInput
              ppgData={ppgData}
              ecgData={ecgData}
              onPPGChange={handlePPGChange}
              onECGChange={handleECGChange}
            />
          </div>

          {/* Controls */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Prediction Controls
            </h2>
            
            {/* Quality metrics toggle */}
            <div className="mb-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={includePPGuality}
                  onChange={(e) => setIncludeQuality(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span className="ml-2 text-gray-700">Include signal quality metrics</span>
              </label>
            </div>

            {/* Action buttons */}
            <div className="flex space-x-4">
              <button
                onClick={handlePredict}
                disabled={!isDataValid || isLoading}
                className={`
                  flex-1 flex items-center justify-center px-6 py-3 rounded-lg font-medium
                  transition-colors duration-200
                  ${isDataValid && !isLoading
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }
                `}
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Predicting...
                  </>
                ) : (
                  <>
                    <ChartBarIcon className="w-5 h-5 mr-2" />
                    Predict BP
                  </>
                )}
              </button>
              
              <button
                onClick={handleClearData}
                className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 
                         rounded-lg font-medium transition-colors duration-200"
              >
                Clear
              </button>
            </div>

            {/* Data validation status */}
            <div className="mt-4 p-3 rounded-lg bg-gray-50">
              <div className="flex items-center">
                {isDataValid ? (
                  <>
                    <CheckCircleIcon className="w-5 h-5 text-green-500 mr-2" />
                    <span className="text-green-700 text-sm">Data is valid and ready for prediction</span>
                  </>
                ) : (
                  <>
                    <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500 mr-2" />
                    <span className="text-yellow-700 text-sm">Please provide valid PPG and ECG signals</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Right Column - Visualization and Results */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="space-y-6"
        >
          {/* Signal Visualization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Signal Visualization
            </h2>
            <SignalVisualization ppgData={ppgData} ecgData={ecgData} />
          </div>

          {/* Quality Assessment */}
          {prediction && prediction.quality_metrics && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <InformationCircleIcon className="w-6 h-6 mr-2 text-blue-600" />
                Signal Quality Assessment
              </h2>
              <QualityAssessment qualityMetrics={prediction.quality_metrics} />
            </div>
          )}

          {/* Prediction Results */}
          {(prediction || error) && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Prediction Results
              </h2>
              <PredictionResults prediction={prediction} error={error} />
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default PredictionInterface;
