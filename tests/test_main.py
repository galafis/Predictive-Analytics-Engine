import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os

# Import the main script to be tested
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main import PredictiveAnalyticsAnalyzer, main, config

class TestPredictiveAnalyticsEngine(unittest.TestCase):

    def setUp(self):
        # Ensure a clean config for each test
        config.model.test_size = 0.2
        config.model.random_state = 42
        config.model.n_estimators = 100
        config.performance.n_jobs = -1
        config.model.verbose = 0
        self.analyzer = PredictiveAnalyticsAnalyzer()

    def test_load_data_default(self):
        self.analyzer.load_data()
        self.assertIsNotNone(self.analyzer.data)
        self.assertEqual(self.analyzer.data.shape, (1000, 4))
        self.assertIn('feature1', self.analyzer.data.columns)
        self.assertIn('target', self.analyzer.data.columns)

    def test_load_data_custom(self):
        custom_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        self.analyzer.load_data(custom_data)
        self.assertIsNotNone(self.analyzer.data)
        self.assertEqual(self.analyzer.data.shape, (3, 3))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_analyze_and_visualize(self, mock_show, mock_savefig):
        self.analyzer.load_data()
        results = self.analyzer.analyze()
        self.assertIsNotNone(results)
        self.assertIn('statistics', results)
        self.assertIn('classification_report', results)
        self.assertIsNotNone(self.analyzer.model)

        self.analyzer.visualize()
        mock_savefig.assert_called_once_with('predictive_analytics_analysis.png', dpi=300, bbox_inches='tight')
        # mock_show.assert_called_once() # Removed plt.show() from main.py

    @patch('main.PredictiveAnalyticsAnalyzer')
    def test_main_function(self, MockPredictiveAnalyticsAnalyzer):
        mock_analyzer_instance = MockPredictiveAnalyticsAnalyzer.return_value
        mock_analyzer_instance.analyze.return_value = {'test_result': 'success'}
        
        main()
        MockPredictiveAnalyticsAnalyzer.assert_called_once()
        mock_analyzer_instance.analyze.assert_called_once()
        mock_analyzer_instance.visualize.assert_called_once()

if __name__ == '__main__':
    unittest.main()
