"""
Predictive-Analytics-Engine
===========================
Demonstracao de pipeline de classificacao com Random Forest, EDA automatizada
e visualizacoes com matplotlib/seaborn. Projeto educacional.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from config.config import get_config

config = get_config()


class PredictiveAnalyticsAnalyzer:
    """Pipeline de classificacao com EDA, Random Forest e visualizacoes."""

    def __init__(self):
        self.config = config
        self.data = None
        self.model = None
        self.results = {}

    def load_data(self, data=None):
        """Carrega dados ou gera dataset sintetico para demonstracao."""
        if data is None:
            np.random.seed(42)
            n = 1000
            f1 = np.random.randn(n)
            f2 = np.random.randn(n)
            f3 = np.random.randn(n)
            target = (f1 + 0.5 * f2 + np.random.randn(n) * 0.3 > 0).astype(int)
            self.data = pd.DataFrame({
                'feature1': f1,
                'feature2': f2,
                'feature3': f3,
                'target': target,
            })
        else:
            self.data = data
        print(f"Data loaded: {self.data.shape}")
        return self.data

    def analyze(self):
        """Treina Random Forest e retorna metricas de classificacao."""
        if self.data is None:
            self.load_data()

        self.results['statistics'] = self.data.describe()

        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
        )

        self.model = RandomForestClassifier(
            n_estimators=self.config.model.n_estimators,
            random_state=self.config.model.random_state,
            n_jobs=self.config.performance.n_jobs,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.results['accuracy'] = accuracy_score(y_test, y_pred)
        self.results['classification_report'] = classification_report(y_test, y_pred)

        print(f"Accuracy: {self.results['accuracy']:.4f}")
        return self.results

    def visualize(self, output_path='predictive_analytics_analysis.png'):
        """Gera graficos EDA e salva em arquivo PNG."""
        if self.data is None:
            self.load_data()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=axes[0, 0])
        axes[0, 0].set_title('Feature Correlations')

        self.data['feature1'].hist(bins=30, alpha=0.7, ax=axes[0, 1])
        axes[0, 1].set_title('Feature 1 Distribution')

        sns.scatterplot(data=self.data, x='feature1', y='feature2', hue='target', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Scatter Plot')

        if self.model is not None:
            importance = pd.DataFrame({
                'feature': self.data.drop('target', axis=1).columns,
                'importance': self.model.feature_importances_,
            }).sort_values('importance', ascending=False)
            sns.barplot(data=importance, x='importance', y='feature', ax=axes[1, 1])
            axes[1, 1].set_title('Feature Importance')
        else:
            axes[1, 1].set_title('Run analyze() first')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualizations saved to '{output_path}'")


def main():
    """Executa pipeline completo: dados sinteticos -> analise -> visualizacao."""
    print("Predictive Analytics Engine - Analysis Pipeline")
    print("=" * 50)
    analyzer = PredictiveAnalyticsAnalyzer()
    analyzer.load_data()
    results = analyzer.analyze()
    analyzer.visualize()
    print("\nClassification Report:")
    print(results['classification_report'])
    print("Analysis completed successfully!")
    return analyzer


if __name__ == "__main__":
    main()
