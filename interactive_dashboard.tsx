import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer
} from 'recharts';
import {
  TrendingUp, Activity, MessageSquare, BarChart3,
  RefreshCw, Send, Trash2
} from 'lucide-react';

const SentimentDashboard = () => {
  const [inputText, setInputText] = useState('');
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [realtimeData, setRealtimeData] = useState([]);
  const [activeTab, setActiveTab] = useState('analyze');

  const generateRealtimeData = useCallback(() => {
    const newData = Array.from({ length: 20 }, (_, i) => ({
      time: new Date(Date.now() - (19 - i) * 60000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      positive: Math.floor(Math.random() * 50) + 30,
      negative: Math.floor(Math.random() * 30) + 10,
      neutral: Math.floor(Math.random() * 40) + 20,
    }));
    setRealtimeData(newData);
  }, []);

  useEffect(() => {
    generateRealtimeData();
    const interval = setInterval(generateRealtimeData, 5000);
    return () => clearInterval(interval);
  }, [generateRealtimeData]);

  const samplePredictions = [
    { text: "I absolutely love this new product!", sentiment: "Positive", confidence: 0.94, model: "ensemble" },
    { text: "This service is terrible and slow", sentiment: "Negative", confidence: 0.87, model: "ensemble" },
  ];

  const handleAnalyze = () => {
    if (!inputText.trim()) return;

    setIsLoading(true);

    setTimeout(() => {
      const sentiment = Math.random() > 0.5 ? 'Positive' : Math.random() > 0.5 ? 'Negative' : 'Neutral';
      const confidence = Math.random() * 0.3 + 0.7; // 0.7 to 1.0

      const newPrediction = {
        text: inputText,
        sentiment,
        confidence,
        model: selectedModel,
        timestamp: new Date().toLocaleString()
      };

      setPredictions(prev => [newPrediction, ...prev.slice(0, 9)]);
      setInputText('');
      setIsLoading(false);
    }, 1500);
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'Positive': return 'text-green-600 bg-green-100';
      case 'Negative': return 'text-red-600 bg-red-100';
      case 'Neutral': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getConfidenceBar = (confidence) => {
    const percentage = Math.round(confidence * 100);
    const color = confidence > 0.8 ? 'bg-green-500' : confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500';

    return (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${color}`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    );
  };

  const sentimentDistribution = [
    { name: 'Positive', value: 45, color: '#10B981' },
    { name: 'Negative', value: 25, color: '#EF4444' },
    { name: 'Neutral', value: 30, color: '#F59E0B' },
  ];

  const modelComparison = [
    { name: 'Naive Bayes', accuracy: 82, precision: 81, recall: 82 },
    { name: 'LSTM', accuracy: 87, precision: 86, recall: 87 },
    { name: 'BERT', accuracy: 91, precision: 90, recall: 91 },
    { name: 'Ensemble', accuracy: 94, precision: 93, recall: 94 },
  ];

  // Colors for Pie chart slices
  const COLORS = sentimentDistribution.map(s => s.color);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🎯 Advanced Sentiment Analysis Dashboard
          </h1>
          <p className="text-gray-600 text-lg">
            Real-time sentiment analysis with ensemble machine learning models
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="flex space-x-1 p-1">
            {[
              { id: 'analyze', label: 'Analyze Text', icon: MessageSquare },
              { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
              { id: 'models', label: 'Model Performance', icon: Activity },
              { id: 'realtime', label: 'Real-time Monitor', icon: TrendingUp }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white shadow-md'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <tab.icon size={18} />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'analyze' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Input Section */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                  <MessageSquare className="mr-2 text-blue-500" />
                  Analyze Text
                </h2>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select Model
                    </label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="ensemble">Ensemble Model (Recommended)</option>
                      <option value="bert">BERT Transformer</option>
                      <option value="lstm">LSTM Neural Network</option>
                      <option value="naive_bayes">Naive Bayes</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Enter Text to Analyze
                    </label>
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder="Type your text, tweet, or review here..."
                      className="w-full p-3 border border-gray-300 rounded-md h-32 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    />
                  </div>

                  <button
                    onClick={handleAnalyze}
                    disabled={!inputText.trim() || isLoading}
                    className="w-full bg-blue-500 text-white py-3 px-4 rounded-md hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors"
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw className="animate-spin" size={18} />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Send size={18} />
                        <span>Analyze Sentiment</span>
                      </>
                    )}
                  </button>

                  <div className="flex space-x-2">
                    <button
                      onClick={() => setInputText(samplePredictions[0].text)}
                      className="flex-1 bg-green-100 text-green-700 py-2 px-3 rounded-md text-sm hover:bg-green-200 transition-colors"
                    >
                      Sample Positive
                    </button>
                    <button
                      onClick={() => setInputText(samplePredictions[1].text)}
                      className="flex-1 bg-red-100 text-red-700 py-2 px-3 rounded-md text-sm hover:bg-red-200 transition-colors"
                    >
                      Sample Negative
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Results Section */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold flex items-center">
                    <Activity className="mr-2 text-green-500" />
                    Recent Predictions
                  </h2>
                  <button
                    onClick={() => setPredictions([])}
                    className="text-gray-500 hover:text-red-500 transition-colors"
                    aria-label="Clear Predictions"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>

                {predictions.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <MessageSquare size={48} className="mx-auto mb-4 opacity-50" />
                    <p>No predictions yet. Enter some text to analyze!</p>
                  </div>
                ) : (
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {predictions.map((pred, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between mb-2">
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(pred.sentiment)}`}>
                            {pred.sentiment}
                          </span>
                          <span className="text-xs text-gray-500">{pred.model}</span>
                        </div>
                        <p className="text-gray-700 mb-3">{pred.text}</p>
                        <div className="flex items-center space-x-3">
                          <span className="text-sm text-gray-600 min-w-fit">
                            Confidence: {Math.round(pred.confidence * 100)}%
                          </span>
                          <div className="flex-1">
                            {getConfidenceBar(pred.confidence)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <div className="bg-white rounded-lg shadow-md p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center text-blue-600">
                <PieChart className="mr-2" /> Sentiment Distribution
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={sentimentDistribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                    labelLine={false}
                  >
                    {sentimentDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center text-indigo-600">
                <BarChart3 className="mr-2" /> Model Accuracy Comparison
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelComparison} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[70, 100]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#2563EB" name="Accuracy (%)" />
                  <Bar dataKey="precision" fill="#10B981" name="Precision (%)" />
                  <Bar dataKey="recall" fill="#F59E0B" name="Recall (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center text-purple-700">
              <Activity className="mr-2" /> Detailed Model Performance Metrics
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-purple-100">
                    <th className="p-3 border border-purple-300">Model</th>
                    <th className="p-3 border border-purple-300">Accuracy (%)</th>
                    <th className="p-3 border border-purple-300">Precision (%)</th>
                    <th className="p-3 border border-purple-300">Recall (%)</th>
                    <th className="p-3 border border-purple-300">F1 Score (%)</th>
                    <th className="p-3 border border-purple-300">Inference Time (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { model: 'Naive Bayes', accuracy: 82, precision: 81, recall: 82, f1: 81.5, time: 15 },
                    { model: 'LSTM', accuracy: 87, precision: 86, recall: 87, f1: 86.5, time: 40 },
                    { model: 'BERT', accuracy: 91, precision: 90, recall: 91, f1: 90.5, time: 80 },
                    { model: 'Ensemble', accuracy: 94, precision: 93, recall: 94, f1: 93.5, time: 100 },
                  ].map((row, idx) => (
                    <tr
                      key={idx}
                      className={idx % 2 === 0 ? 'bg-white' : 'bg-purple-50'}
                    >
                      <td className="p-3 border border-purple-300 font-semibold">{row.model}</td>
                      <td className="p-3 border border-purple-300">{row.accuracy}</td>
                      <td className="p-3 border border-purple-300">{row.precision}</td>
                      <td className="p-3 border border-purple-300">{row.recall}</td>
                      <td className="p-3 border border-purple-300">{row.f1}</td>
                      <td className="p-3 border border-purple-300">{row.time}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'realtime' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center text-teal-600">
              <TrendingUp className="mr-2" /> Real-time Sentiment Monitoring
            </h2>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={realtimeData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Legend verticalAlign="top" height={36} />
                <Line type="monotone" dataKey="positive" stroke="#10B981" strokeWidth={2} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="negative" stroke="#EF4444" strokeWidth={2} />
                <Line type="monotone" dataKey="neutral" stroke="#F59E0B" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-2 text-gray-600 text-sm italic">
              Data updates every 5 seconds, simulating live social media sentiment trends.
            </p>
          </div>
        )}

      </div>
    </div>
  );
};

export default SentimentDashboard;
