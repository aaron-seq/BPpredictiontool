/**
 * Advanced Blood Pressure Prediction Tool - Main Application
 */
import React, { useState, useCallback, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import Header from './components/Layout/Header';
import Footer from './components/Layout/Footer';
import PredictionInterface from './components/Prediction/PredictionInterface';
import DataVisualization from './components/Visualization/DataVisualization';
import HealthMonitor from './components/HealthMonitor/HealthMonitor';
import AboutPage from './components/Pages/AboutPage';
import HelpPage from './components/Pages/HelpPage';

// Context
import { AppContextProvider } from './context/AppContext';

// Styles
import './styles/globals.css';

const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [apiHealth, setApiHealth] = useState(null);

  // Check API health on app load
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/health`);
        const healthData = await response.json();
        setApiHealth(healthData);
      } catch (error) {
        console.error('Failed to check API health:', error);
        setApiHealth({ status: 'unhealthy', error: error.message });
      } finally {
        setIsLoading(false);
      }
    };

    checkApiHealth();
  }, []);

  const pageVariants = {
    initial: {
      opacity: 0,
      y: 20
    },
    in: {
      opacity: 1,
      y: 0
    },
    out: {
      opacity: 0,
      y: -20
    }
  };

  const pageTransition = {
    type: 'tween',
    ease: 'anticipate',
    duration: 0.3
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <motion.div
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading Blood Pressure Prediction Tool...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <AppContextProvider>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
          <Header apiHealth={apiHealth} />
          
          <main className="container mx-auto px-4 py-8">
            <AnimatePresence mode="wait">
              <Routes>
                <Route
                  path="/"
                  element={
                    <motion.div
                      key="prediction"
                      initial="initial"
                      animate="in"
                      exit="out"
                      variants={pageVariants}
                      transition={pageTransition}
                    >
                      <PredictionInterface />
                    </motion.div>
                  }
                />
                <Route
                  path="/visualization"
                  element={
                    <motion.div
                      key="visualization"
                      initial="initial"
                      animate="in"
                      exit="out"
                      variants={pageVariants}
                      transition={pageTransition}
                    >
                      <DataVisualization />
                    </motion.div>
                  }
                />
                <Route
                  path="/monitor"
                  element={
                    <motion.div
                      key="monitor"
                      initial="initial"
                      animate="in"
                      exit="out"
                      variants={pageVariants}
                      transition={pageTransition}
                    >
                      <HealthMonitor />
                    </motion.div>
                  }
                />
                <Route
                  path="/about"
                  element={
                    <motion.div
                      key="about"
                      initial="initial"
                      animate="in"
                      exit="out"
                      variants={pageVariants}
                      transition={pageTransition}
                    >
                      <AboutPage />
                    </motion.div>
                  }
                />
                <Route
                  path="/help"
                  element={
                    <motion.div
                      key="help"
                      initial="initial"
                      animate="in"
                      exit="out"
                      variants={pageVariants}
                      transition={pageTransition}
                    >
                      <HelpPage />
                    </motion.div>
                  }
                />
              </Routes>
            </AnimatePresence>
          </main>

          <Footer />
          
          {/* Toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                style: {
                  background: '#10b981',
                },
              },
              error: {
                style: {
                  background: '#ef4444',
                },
              },
            }}
          />
        </div>
      </Router>
    </AppContextProvider>
  );
};

export default App;
