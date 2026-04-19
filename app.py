import os
import io
import base64
import json
import traceback

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import type_of_target

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

import joblib

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'data'
MODEL_FOLDER  = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def detect_problem(y):
    t = type_of_target(y)
    return 'classification' if t in ['binary', 'multiclass'] else 'regression'


def preprocess(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.fillna(X.mean(numeric_only=True))
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, list(df.drop(columns=[target_col]).columns)


@app.route('/api/columns', methods=['POST'])
def get_columns():
    try:
        f = request.files.get('file')
        if not f:
            return jsonify({'error': 'No file'}), 400
        df = pd.read_csv(f)
        return jsonify({'columns': list(df.columns), 'rows': len(df), 'preview': df.head(5).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train():
    try:
        f = request.files.get('file')
        target_col = request.form.get('target_column')
        if not f or not target_col:
            return jsonify({'error': 'File and target_column required'}), 400

        df = pd.read_csv(f)
        if target_col not in df.columns:
            return jsonify({'error': f'Column "{target_col}" not found'}), 400

        problem_type = detect_problem(df[target_col])
        X, y, feature_names = preprocess(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results     = {}
        plots       = {}
        all_reports = {}

        # ── CLASSIFICATION ──────────────────────────────────────────────────
        if problem_type == 'classification':
            models = {
                'Random Forest': (RandomForestClassifier(),  {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
                'SVM':           (SVC(),                     {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
                'Logistic Reg':  (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
            }
            for name, (model, params) in models.items():
                grid = GridSearchCV(model, params, cv=3, scoring='accuracy')
                grid.fit(X_train, y_train)
                best  = grid.best_estimator_
                preds = best.predict(X_test)
                acc   = accuracy_score(y_test, preds)
                results[name] = {'score': round(acc, 4), 'best_params': grid.best_params_, 'model': best}
                all_reports[name] = classification_report(y_test, preds, output_dict=True)

            best_name  = max(results, key=lambda x: results[x]['score'])
            best_model = results[best_name]['model']
            best_preds = best_model.predict(X_test)

            # Confusion matrix
            cm  = confusion_matrix(y_test, best_preds)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0f0f13')
            ax.set_facecolor('#0f0f13')
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax,
                        linewidths=0.5, linecolor='#1e1e2e',
                        annot_kws={'color': 'white', 'fontsize': 11})
            ax.set_xlabel('Predicted', color='#a0a0b0', fontsize=11)
            ax.set_ylabel('Actual',    color='#a0a0b0', fontsize=11)
            ax.tick_params(colors='#a0a0b0')
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            plots['confusion_matrix'] = fig_to_base64(fig)

            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                fi  = best_model.feature_importances_
                idx = np.argsort(fi)[::-1][:10]
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0f0f13')
                ax.set_facecolor('#0f0f13')
                bars = ax.barh([feature_names[i] for i in idx[::-1]], fi[idx[::-1]], color='#c084fc')
                ax.tick_params(colors='#a0a0b0')
                ax.set_xlabel('Importance', color='#a0a0b0')
                for spine in ax.spines.values(): spine.set_color('#2a2a3e')
                plt.tight_layout()
                plots['feature_importance'] = fig_to_base64(fig)

            metric_label = 'Accuracy'

        # ── REGRESSION ──────────────────────────────────────────────────────
        else:
            models = {
                'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
                'SVR':           (SVR(),                   {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
                'Linear Reg':    (LinearRegression(),      {}),
            }
            for name, (model, params) in models.items():
                if params:
                    grid = GridSearchCV(model, params, cv=3, scoring='r2')
                    grid.fit(X_train, y_train)
                    best = grid.best_estimator_
                    bp   = grid.best_params_
                else:
                    model.fit(X_train, y_train)
                    best, bp = model, {}
                preds = best.predict(X_test)
                score = r2_score(y_test, preds)
                results[name] = {'score': round(score, 4), 'best_params': bp, 'model': best}

            best_name  = max(results, key=lambda x: results[x]['score'])
            best_model = results[best_name]['model']
            best_preds = best_model.predict(X_test)

            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0f0f13')
            ax.set_facecolor('#0f0f13')
            ax.scatter(y_test, best_preds, color='#c084fc', alpha=0.6, s=30)
            mn, mx = float(min(y_test)), float(max(y_test))
            ax.plot([mn, mx], [mn, mx], color='#f472b6', linewidth=1.5, linestyle='--')
            ax.set_xlabel('Actual',    color='#a0a0b0')
            ax.set_ylabel('Predicted', color='#a0a0b0')
            ax.tick_params(colors='#a0a0b0')
            for spine in ax.spines.values(): spine.set_color('#2a2a3e')
            plt.tight_layout()
            plots['actual_vs_predicted'] = fig_to_base64(fig)

            if hasattr(best_model, 'feature_importances_'):
                fi  = best_model.feature_importances_
                idx = np.argsort(fi)[::-1][:10]
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0f0f13')
                ax.set_facecolor('#0f0f13')
                ax.barh([feature_names[i] for i in idx[::-1]], fi[idx[::-1]], color='#c084fc')
                ax.tick_params(colors='#a0a0b0')
                ax.set_xlabel('Importance', color='#a0a0b0')
                for spine in ax.spines.values(): spine.set_color('#2a2a3e')
                plt.tight_layout()
                plots['feature_importance'] = fig_to_base64(fig)

            metric_label = 'R² Score'

        # ── Model comparison bar chart ───────────────────────────────────────
        names  = list(results.keys())
        scores = [results[n]['score'] for n in names]
        colors = ['#c084fc' if n == best_name else '#4a3f6b' for n in names]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#0f0f13')
        ax.set_facecolor('#0f0f13')
        bars = ax.bar(names, scores, color=colors, width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric_label, color='#a0a0b0')
        ax.tick_params(colors='#a0a0b0', axis='both')
        for spine in ax.spines.values(): spine.set_color('#2a2a3e')
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', color='white', fontsize=10)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plots['model_comparison'] = fig_to_base64(fig)

        # Save best model
        model_path = os.path.join(MODEL_FOLDER, 'best_model.pkl')
        joblib.dump(best_model, model_path)

        # Build clean results (remove model objects)
        clean_results = {k: {'score': v['score'], 'best_params': v['best_params']} for k, v in results.items()}

        return jsonify({
            'problem_type':  problem_type,
            'metric_label':  metric_label,
            'best_model':    best_name,
            'best_score':    results[best_name]['score'],
            'all_results':   clean_results,
            'plots':         plots,
            'reports':       all_reports,
            'dataset_info':  {'rows': len(df), 'features': len(feature_names), 'target': target_col},
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
from flask import send_file

@app.route('/api/download-model', methods=['GET'])
def download_model():
    model_path = os.path.join(MODEL_FOLDER, 'best_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'No model trained yet'}), 404
    return send_file(model_path, as_attachment=True, download_name='best_model.pkl')


if __name__ == '__main__':
    app.run(debug=True, port=5000)