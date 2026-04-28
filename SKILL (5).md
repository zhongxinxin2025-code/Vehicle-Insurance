{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "de7bb178",
      "metadata": {
        "id": "de7bb178"
      },
      "source": [
        "# Section 0 — Setup and Installs"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "id": "39c0ee86",
      "metadata": {
        "id": "39c0ee86",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9ea86b55-c13f-486f-fff9-f88c5ebf830d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m88.5/88.5 kB\u001b[0m \u001b[31m2.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m506.8/506.8 kB\u001b[0m \u001b[31m13.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "!pip install -q langgraph langchain langchain-openai gradio imbalanced-learn xgboost scikit-learn pandas numpy matplotlib seaborn"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "id": "28821724",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "28821724",
        "outputId": "80233e97-4bc1-4b43-a17b-c6b0678f3d66"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter your OpenAI API key: ··········\n",
            "API key loaded successfully.\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import getpass\n",
        "\n",
        "# Works in both Colab browser UI and Colab via VS Code\n",
        "if 'OPENAI_API_KEY' not in os.environ:\n",
        "    os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API key: ')\n",
        "\n",
        "openai_key = os.environ['OPENAI_API_KEY']\n",
        "print('API key loaded successfully.')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "7490a32f",
      "metadata": {
        "id": "7490a32f"
      },
      "source": [
        "# Section 1 — Preprocess-Data Agent Node"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "id": "a8e1eedd",
      "metadata": {
        "lines_to_next_cell": 1,
        "id": "a8e1eedd"
      },
      "outputs": [],
      "source": [
        "from typing import TypedDict, Optional\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from langgraph.graph import StateGraph, END"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "id": "92ea0522",
      "metadata": {
        "lines_to_next_cell": 1,
        "id": "92ea0522"
      },
      "outputs": [],
      "source": [
        "from typing import TypedDict, Optional, Any\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "class AgentState(TypedDict):\n",
        "\n",
        "    # --- preprocess-data outputs ---\n",
        "    raw_data_path:    str\n",
        "    X_train:          pd.DataFrame\n",
        "    y_train:          pd.Series\n",
        "    X_val:            pd.DataFrame\n",
        "    y_val:            pd.Series\n",
        "    X_test:           pd.DataFrame\n",
        "    y_test:           pd.Series\n",
        "    encoder:          Any                  # fitted OneHotEncoder\n",
        "    feature_names:    list[str]\n",
        "\n",
        "    # --- train-models outputs ---\n",
        "    trained_models:   dict[str, Any]       # {\"logistic\": model, \"random_forest\": model, \"xgboost\": model}\n",
        "\n",
        "    # --- evaluate-models outputs ---\n",
        "    eval_results:     dict[str, dict]      # {\"logistic\": {\"accuracy\": 0.9, \"recall\": 0.8 ...}, ...}\n",
        "\n",
        "    # --- select-model outputs ---\n",
        "    selected_model:   Any                  # the winning model object\n",
        "    selected_model_name: str              # e.g. \"xgboost\"\n",
        "    selection_justification: str          # plain English explanation\n",
        "\n",
        "    # --- run-inference outputs ---\n",
        "    input_claim:      dict                 # raw feature values from Gradio form\n",
        "    fraud_probability: float\n",
        "    predicted_label:  int                  # 0 or 1\n",
        "\n",
        "    # --- fraud-decision-engine outputs ---\n",
        "    risk_level:       str                  # \"Low\", \"Medium\", or \"High\"\n",
        "    decision_output:  str\n",
        "\n",
        "    conf_matrices:    dict   # confusion matrix per model\n",
        "    roc_curves:       dict   # ROC curve data per model\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "id": "8f238f0a",
      "metadata": {
        "lines_to_next_cell": 1,
        "id": "8f238f0a"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from imblearn.over_sampling import SMOTE\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "def preprocess_data(state: AgentState) -> AgentState:\n",
        "    \"\"\"\n",
        "    LangGraph node: preprocesses raw vehicle insurance data.\n",
        "\n",
        "    Steps:\n",
        "      1. Load CSV from raw_data_path.\n",
        "      2. Drop PolicyNumber and RepNumber (identifiers, not predictive).\n",
        "      3. Stratified 70/15/15 train/val/test split.\n",
        "      4. Fit OneHotEncoder on training categoricals; transform val and test.\n",
        "      5. Apply SMOTE to training set only to handle class imbalance.\n",
        "      6. Store all splits, fitted encoder, and feature names in agent state.\n",
        "    \"\"\"\n",
        "    df = pd.read_csv(state['raw_data_path'])\n",
        "\n",
        "    # ── Step 1: Drop identifier columns ──────────────────────────────────────\n",
        "    cols_to_drop = [c for c in ['PolicyNumber', 'RepNumber'] if c in df.columns]\n",
        "    df.drop(columns=cols_to_drop, inplace=True)\n",
        "    print(f'[preprocess_data] Dropped columns: {cols_to_drop}')\n",
        "\n",
        "    # ── Step 2: Separate target ───────────────────────────────────────────────\n",
        "    target = 'FraudFound_P'\n",
        "    X = df.drop(columns=[target])\n",
        "    y = df[target]\n",
        "\n",
        "    # ── Step 3: Stratified 70 / 15 / 15 split (before encoding to avoid leakage) ──\n",
        "    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(\n",
        "        X, y, test_size=0.30, stratify=y, random_state=42\n",
        "    )\n",
        "    X_val_raw, X_test_raw, y_val, y_test = train_test_split(\n",
        "        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42\n",
        "    )\n",
        "    print(f'[preprocess_data] Split sizes — train: {len(X_train_raw)}, '\n",
        "          f'val: {len(X_val_raw)}, test: {len(X_test_raw)}')\n",
        "\n",
        "    # ── Step 4: Fit OneHotEncoder on train, apply to val and test ─────────────\n",
        "    cat_cols = X_train_raw.select_dtypes(include=['object', 'string']).columns.tolist()\n",
        "    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()\n",
        "    print(f'[preprocess_data] Encoding {len(cat_cols)} categorical columns')\n",
        "\n",
        "    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n",
        "    encoder.fit(X_train_raw[cat_cols])\n",
        "\n",
        "    def apply_encoding(X_raw):\n",
        "        encoded = encoder.transform(X_raw[cat_cols])\n",
        "        encoded_df = pd.DataFrame(\n",
        "            encoded,\n",
        "            columns=encoder.get_feature_names_out(cat_cols),\n",
        "            index=X_raw.index\n",
        "        )\n",
        "        return pd.concat([X_raw[num_cols].reset_index(drop=True),\n",
        "                          encoded_df.reset_index(drop=True)], axis=1)\n",
        "\n",
        "    X_train_enc = apply_encoding(X_train_raw)\n",
        "    X_val_enc   = apply_encoding(X_val_raw)\n",
        "    X_test_enc  = apply_encoding(X_test_raw)\n",
        "    feature_names = X_train_enc.columns.tolist()\n",
        "    print(f'[preprocess_data] Total features after encoding: {len(feature_names)}')\n",
        "\n",
        "    # ── Step 5: SMOTE on training set only ───────────────────────────────────\n",
        "    print(f'[preprocess_data] Train class distribution before SMOTE: '\n",
        "          f'{y_train_raw.value_counts().to_dict()}')\n",
        "    smote = SMOTE(random_state=42)\n",
        "    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train_raw)\n",
        "    X_train_res = pd.DataFrame(X_train_res, columns=feature_names)\n",
        "    y_train_res = pd.Series(y_train_res, name=target)\n",
        "    print(f'[preprocess_data] Train class distribution after SMOTE: '\n",
        "          f'{y_train_res.value_counts().to_dict()}')\n",
        "\n",
        "    return {\n",
        "        **state,\n",
        "        'X_train': X_train_res,\n",
        "        'y_train': y_train_res,\n",
        "        'X_val':   X_val_enc.reset_index(drop=True),\n",
        "        'y_val':   y_val.reset_index(drop=True),\n",
        "        'X_test':  X_test_enc.reset_index(drop=True),\n",
        "        'y_test':  y_test.reset_index(drop=True),\n",
        "        'encoder': encoder,\n",
        "        'feature_names': feature_names,\n",
        "    }\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "id": "dcf272ca",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dcf272ca",
        "outputId": "e9dba29d-a061-4285-ad38-4e9ff3f5a8f3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "[preprocess_data] Dropped columns: ['PolicyNumber', 'RepNumber']\n",
            "[preprocess_data] Split sizes — train: 10794, val: 2313, test: 2313\n",
            "[preprocess_data] Encoding 24 categorical columns\n",
            "[preprocess_data] Total features after encoding: 146\n",
            "[preprocess_data] Train class distribution before SMOTE: {0: 10148, 1: 646}\n",
            "[preprocess_data] Train class distribution after SMOTE: {0: 10148, 1: 10148}\n",
            "\n",
            "── Verification ──────────────────────────────────────\n",
            "X_train shape : (20296, 146)\n",
            "X_val shape   : (2313, 146)\n",
            "X_test shape  : (2313, 146)\n",
            "y_train counts: {0: 10148, 1: 10148}\n",
            "y_val counts  : {0: 2174, 1: 139}\n",
            "y_test counts : {0: 2175, 1: 138}\n",
            "Features      : 146\n"
          ]
        }
      ],
      "source": [
        "# In Colab, either upload manually or mount Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Then update raw_data_path in initial_state to point to the correct Colab path\n",
        "initial_state: AgentState = {\n",
        "    'raw_data_path': '/content/drive/MyDrive/fraud_oracle.csv',\n",
        "    # ... rest of the fields\n",
        "}\n",
        "# ── Quick smoke-test: run the node standalone ─────────────────────────────────\n",
        "# ── Quick smoke-test: run the node standalone ─────────────────────────────────\n",
        "initial_state: AgentState = {\n",
        "    'raw_data_path': '/content/drive/MyDrive/Colab Notebooks/fraud_oracle.csv',\n",
        "    'X_train': None,\n",
        "    'X_val': None,\n",
        "    'X_test': None,\n",
        "    'y_train': None,\n",
        "    'y_val': None,\n",
        "    'y_test': None,\n",
        "    'encoder': None,\n",
        "    'feature_names': None,\n",
        "    'trained_models': None,\n",
        "    'eval_results': None,\n",
        "    'selected_model': None,\n",
        "    'selected_model_name': None,\n",
        "    'selection_justification': None,\n",
        "    'input_claim': None,\n",
        "    'fraud_probability': None,\n",
        "    'predicted_label': None,\n",
        "    'risk_level': None,\n",
        "    'decision_output': None,\n",
        "    'conf_matrices': None,\n",
        "    'roc_curves': None,\n",
        "}\n",
        "\n",
        "result_state = preprocess_data(initial_state)\n",
        "\n",
        "print()\n",
        "print('── Verification ──────────────────────────────────────')\n",
        "print(f'X_train shape : {result_state[\"X_train\"].shape}')\n",
        "print(f'X_val shape   : {result_state[\"X_val\"].shape}')\n",
        "print(f'X_test shape  : {result_state[\"X_test\"].shape}')\n",
        "print(f'y_train counts: {result_state[\"y_train\"].value_counts().to_dict()}')\n",
        "print(f'y_val counts  : {result_state[\"y_val\"].value_counts().to_dict()}')\n",
        "print(f'y_test counts : {result_state[\"y_test\"].value_counts().to_dict()}')\n",
        "print(f'Features      : {len(result_state[\"feature_names\"])}')"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "c3e1dc6d",
      "metadata": {
        "id": "c3e1dc6d"
      },
      "source": [
        "# Section 2 — train_model Agent Node (Hrithik)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "id": "8d826d90",
      "metadata": {
        "id": "8d826d90"
      },
      "outputs": [],
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from xgboost import XGBClassifier\n",
        "\n",
        "def train_models(state: AgentState) -> AgentState:\n",
        "    X_train = state['X_train']\n",
        "    y_train = state['y_train']\n",
        "\n",
        "    neg = (y_train == 0).sum()\n",
        "    pos = (y_train == 1).sum()\n",
        "    scale = neg / pos\n",
        "\n",
        "    print(f'[train_models] Training on {X_train.shape[0]} samples, {X_train.shape[1]} features')\n",
        "\n",
        "    # Model A: Logistic Regression\n",
        "    print('[train_models] Fitting Logistic Regression...')\n",
        "    lr = LogisticRegression(\n",
        "        class_weight='balanced',\n",
        "        max_iter=3000,\n",
        "        solver='saga',\n",
        "        random_state=42\n",
        "    )\n",
        "    lr.fit(X_train, y_train)\n",
        "\n",
        "    # Model B: Random Forest\n",
        "    print('[train_models] Fitting Random Forest...')\n",
        "    rf = RandomForestClassifier(\n",
        "        n_estimators=100,\n",
        "        class_weight='balanced',\n",
        "        random_state=42,\n",
        "        n_jobs=-1\n",
        "    )\n",
        "    rf.fit(X_train, y_train)\n",
        "\n",
        "    # Model C: XGBoost\n",
        "    print('[train_models] Fitting XGBoost...')\n",
        "    xgb = XGBClassifier(\n",
        "        scale_pos_weight=scale,\n",
        "        n_estimators=100,\n",
        "        random_state=42,\n",
        "        eval_metric='logloss',\n",
        "        verbosity=0\n",
        "    )\n",
        "    xgb.fit(X_train, y_train)\n",
        "\n",
        "    trained_models = {\n",
        "        'logistic_regression': lr,\n",
        "        'random_forest':       rf,\n",
        "        'xgboost':             xgb\n",
        "    }\n",
        "\n",
        "    print(f'[train_models] All models trained: {list(trained_models.keys())}')\n",
        "\n",
        "    return {\n",
        "        **state,\n",
        "        'trained_models': trained_models\n",
        "    }"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "id": "428e11f9",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "428e11f9",
        "outputId": "5d07af76-0714-4b76-e1ec-71fb0b32124b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[train_models] Training on 20296 samples, 146 features\n",
            "[train_models] Fitting Logistic Regression...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[train_models] Fitting Random Forest...\n",
            "[train_models] Fitting XGBoost...\n",
            "[train_models] All models trained: ['logistic_regression', 'random_forest', 'xgboost']\n",
            "\n",
            "── Verification ──────────────────────────────────────\n",
            "logistic_regression: LogisticRegression fitted = True\n",
            "random_forest: RandomForestClassifier fitted = True\n",
            "xgboost: XGBClassifier fitted = True\n"
          ]
        }
      ],
      "source": [
        "result_state = train_models(result_state)\n",
        "\n",
        "print()\n",
        "print('── Verification ──────────────────────────────────────')\n",
        "for name, model in result_state['trained_models'].items():\n",
        "    print(f'{name}: {type(model).__name__} fitted = {hasattr(model, \"classes_\")}')"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "59a1938c",
      "metadata": {
        "id": "59a1938c"
      },
      "source": [
        "# Section 3 — Evaluate-Model Agent Node"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "id": "bbecfa93",
      "metadata": {
        "id": "bbecfa93"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score,\n",
        "    f1_score, roc_auc_score, confusion_matrix, roc_curve\n",
        ")\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "def evaluate_models(state: AgentState) -> AgentState:\n",
        "    \"\"\"\n",
        "    LangGraph node: evaluates all trained models on the validation set.\n",
        "\n",
        "    Steps:\n",
        "      1. Retrieve X_val, y_val and trained_models from agent state.\n",
        "      2. For each model, compute Accuracy, Precision, Recall, F1, AUC-ROC.\n",
        "      3. Generate a Confusion Matrix and ROC Curve per model.\n",
        "      4. Store all results in agent state under eval_results.\n",
        "    \"\"\"\n",
        "    X_val          = state['X_val']\n",
        "    y_val          = state['y_val']\n",
        "    trained_models = state['trained_models']\n",
        "\n",
        "    eval_results   = {}\n",
        "    conf_matrices  = {}\n",
        "    roc_curves     = {}\n",
        "\n",
        "    for model_name, model in trained_models.items():\n",
        "        print(f'[evaluate_models] Evaluating {model_name}...')\n",
        "\n",
        "        y_pred      = model.predict(X_val)\n",
        "        y_prob      = model.predict_proba(X_val)[:, 1]\n",
        "\n",
        "        # ── Core metrics ──────────────────────────────────────────────────────\n",
        "        metrics = {\n",
        "            'accuracy' : accuracy_score(y_val, y_pred),\n",
        "            'precision': precision_score(y_val, y_pred, zero_division=0),\n",
        "            'recall'   : recall_score(y_val, y_pred, zero_division=0),\n",
        "            'f1'       : f1_score(y_val, y_pred, zero_division=0),\n",
        "            'auc_roc'  : roc_auc_score(y_val, y_prob),\n",
        "        }\n",
        "        eval_results[model_name] = metrics\n",
        "        print(f'  accuracy={metrics[\"accuracy\"]:.4f}  precision={metrics[\"precision\"]:.4f}  '\n",
        "              f'recall={metrics[\"recall\"]:.4f}  f1={metrics[\"f1\"]:.4f}  '\n",
        "              f'auc_roc={metrics[\"auc_roc\"]:.4f}')\n",
        "\n",
        "        # ── Confusion matrix ──────────────────────────────────────────────────\n",
        "        conf_matrices[model_name] = confusion_matrix(y_val, y_pred)\n",
        "\n",
        "        # ── ROC curve data ────────────────────────────────────────────────────\n",
        "        fpr, tpr, thresholds = roc_curve(y_val, y_prob)\n",
        "        roc_curves[model_name] = (fpr, tpr, thresholds)\n",
        "\n",
        "    # ── Plot confusion matrices ────────────────────────────────────────────────\n",
        "    fig, axes = plt.subplots(1, len(trained_models), figsize=(6 * len(trained_models), 5))\n",
        "    if len(trained_models) == 1:\n",
        "        axes = [axes]\n",
        "    for ax, (model_name, cm) in zip(axes, conf_matrices.items()):\n",
        "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n",
        "                    xticklabels=['Not Fraud', 'Fraud'],\n",
        "                    yticklabels=['Not Fraud', 'Fraud'])\n",
        "        ax.set_title(f'Confusion Matrix\\n{model_name}')\n",
        "        ax.set_xlabel('Predicted')\n",
        "        ax.set_ylabel('Actual')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "    # ── Plot ROC curves ───────────────────────────────────────────────────────\n",
        "    plt.figure(figsize=(8, 6))\n",
        "    for model_name, (fpr, tpr, _) in roc_curves.items():\n",
        "        auc = eval_results[model_name]['auc_roc']\n",
        "        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')\n",
        "    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')\n",
        "    plt.xlabel('False Positive Rate')\n",
        "    plt.ylabel('True Positive Rate')\n",
        "    plt.title('ROC Curves — All Models')\n",
        "    plt.legend(loc='lower right')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "    print('[evaluate_models] Evaluation complete.')\n",
        "\n",
        "    return {\n",
        "        **state,\n",
        "        'eval_results':    eval_results,\n",
        "        'conf_matrices':   conf_matrices,\n",
        "        'roc_curves':      roc_curves,\n",
        "    }"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "id": "5d3672e4",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "5d3672e4",
        "outputId": "1745ac08-8d41-49dc-e425-0949c3b7841f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[evaluate_models] Evaluating logistic_regression...\n",
            "  accuracy=0.6775  precision=0.1330  recall=0.7914  f1=0.2277  auc_roc=0.7824\n",
            "[evaluate_models] Evaluating random_forest...\n",
            "  accuracy=0.9395  precision=0.3333  recall=0.0072  f1=0.0141  auc_roc=0.8378\n",
            "[evaluate_models] Evaluating xgboost...\n",
            "  accuracy=0.9377  precision=0.4194  recall=0.0935  f1=0.1529  auc_roc=0.8565\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1800x500 with 6 Axes>"
            ],
            "image/png": "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ioqKcnY3bujDDz9U5cqVVbhwYfn7+zu7OwAAALct837Zm2++6eyuAEC+wWAdkA/NnTvXYQN8AAAAKJj+/PNP9e7dW+XLl9c777yjBQsWOLtLWRw9elRRUVGKj493dlcAAAAc4ocffsjXk7kAuKZCzu4A4MpCQ0N16dIlFS5cOEfHzZ07VyVKlFDv3r2typs1a6ZLly7Jw8PDgb28c73zzjvKyMhwdjcAAACytW7dOmVkZGjmzJmqUKGCs7uTraNHj2rcuHEqW7asateu7ezuAAAA3LYffvhBc+bMYcAOQJ7iyTogF5lMJnl6esrd3d0h7bm5ucnT01Nubrn/o3v58uU8G8jKyMjQ5cuX8+Rc1ypcuLDMZnOenxcAANxYcnKys7uQb5w4cUKSHLr85cWLFx3WFgAAAADAMRisA64zd+5cVatWTWazWSEhIRo4cKDOnj2bpd6cOXNUrlw5eXl5qUGDBvrll1/UokULtWjRwlInu3fWJSYmqk+fPrr77rtlNptVqlQpderUyfLevLJly2rXrl36+eefZTKZZDKZLG3e6J11mzZtUvv27VWsWDF5e3urZs2amjlzps3XnNnuJ598oldeeUV33XWXihQpoqSkJEv77dq1k5+fn4oUKaLmzZvrt99+y7ad+vXry9PTU+XLl9fbb7+tqKgomUwmq3omk0mDBg3SkiVLLN/1ihUrJEn//POP+vbtq6CgIJnNZlWrVk3vv/9+lnPNnj1b1apVU5EiRVSsWDHVr19fS5cutew/f/68hg4dqrJly8psNiswMFAPPPCAtm7daqmT3TvrkpOT9cILL6h06dIym82qVKmS3nzzTRmGke01LF++XNWrV7f0NfM6AADArWXmhN27d6tHjx4qVqyYmjRpou3bt6t3794qV66cPD09FRwcrL59++rUqVPZHr9//3717t1b/v7+8vPzU58+fbIMSqWkpGjYsGEqWbKkfHx89NBDD+nvv//Otl/btm1TRESEfH19VbRoUbVu3VobN260qpP57uNff/1VQ4YMUcmSJeXv76+nn35aV65c0dmzZ9WrVy8VK1ZMxYoV08iRI7PkiZspW7asxo4dK0kqWbJklnfr2ZJZW7RooerVqysuLk7NmjVTkSJF9J///MfyfYwdO1YVKlSQ2WxW6dKlNXLkSKWkpFi1ERMToyZNmsjf319FixZVpUqVLG2sW7dO9957rySpT58+luzKcu4AALiWS5cuqXLlyqpcubIuXbpkKT99+rRKlSql+++/X+np6ZKkzz//XFWrVpWnp6eqV6+uZcuWZXv/JdP06dMVGhoqLy8vNW/eXDt37sxSZ82aNWratKm8vb3l7++vTp06ac+ePVnq2ZLhUlNTNW7cOFWsWFGenp4qXry4mjRpopiYGElX7xXNmTNHkizZ5vr7WgCQG1gGE7hGVFSUxo0bp/DwcD377LPau3ev5s2bp82bN+u3336zLGc5b948DRo0SE2bNtWwYcN06NAhde7cWcWKFdPdd99903N07dpVu3bt0uDBg1W2bFmdOHFCMTExOnLkiMqWLasZM2Zo8ODBKlq0qF5++WVJUlBQ0A3bi4mJ0YMPPqhSpUrp+eefV3BwsPbs2aPvvvtOzz//fI6u/7XXXpOHh4defPFFpaSkyMPDQ2vWrFFERITq1aunsWPHys3NTQsXLlSrVq30yy+/qEGDBpKuBqJ27dqpVKlSGjdunNLT0zV+/HiVLFky23OtWbNGn332mQYNGqQSJUqobNmyOn78uO677z7LQFjJkiX1448/ql+/fkpKStLQoUMlXV2+csiQIXrkkUf0/PPP6/Lly9q+fbs2bdqkHj16SJKeeeYZffHFFxo0aJCqVq2qU6dO6ddff9WePXtUt27dbPtkGIYeeughrV27Vv369VPt2rX1008/acSIEfrnn380ffp0q/q//vqrvvrqKz333HPy8fHRrFmz1LVrVx05ckTFixfP0XcPAMCd7NFHH1XFihU1ceJEGYahmJgYHTx4UH369FFwcLB27dqlBQsWaNeuXdq4cWOWGybdunVTWFiYoqOjtXXrVr377rsKDAzUpEmTLHWeeuopffTRR+rRo4fuv/9+rVmzRh06dMjSl127dqlp06by9fXVyJEjVbhwYb399ttq0aKFfv75ZzVs2NCq/uDBgxUcHKxx48Zp48aNWrBggfz9/bV+/XqVKVNGEydO1A8//KApU6aoevXq6tWrl03fyYwZM/TBBx9o2bJlmjdvnooWLaqaNWtKsj2zStKpU6cUERGh7t2764knnlBQUJAyMjL00EMP6ddff9WAAQNUpUoV7dixQ9OnT9dff/2l5cuXW76LBx98UDVr1tT48eNlNpu1f/9+y6StKlWqaPz48RozZowGDBigpk2b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OmW1z8PDQ1WrVpVhGEpNTVV6enqWJbICAwMVEhJi9T1fr3379kpPT9dbb71lVT59+nSZTCaHPZEHAABuLPNpreuzzIwZM+xuM/Nv+KxZs27apru7u9q0aaOvv/7asvySJB0/flxLly5VkyZN5Ovra3c/HMHWzHoz3bp10z///KN33nkny75Lly4pOTlZ0tXlSK+XudpCZqby9vaWJJuyKwAAKLhSU1PVu3dvhYSEaObMmVq0aJGOHz+uYcOGSZJCQkJUvXp1ffDBB1b3kH7++Wft2LEj2zaXL1+uf/75x/L5999/16ZNmyzZrVSpUqpdu7YWL15slTV27typlStXqn379pJyluGuv6dUtGhRVahQIct9OYl8AyBv8WQd8P9Kliyp0aNHa9y4cWrXrp0eeugh7d27V3PnztW9996rJ554QtLVgaGoqCgNHjxYrVq1Urdu3XTo0CEtWrRI5cuXv+nsm7/++kutW7dWt27dVLVqVRUqVEjLli3T8ePH1b17d0u9evXqad68eZowYYIqVKigwMBAtWrVKkt7bm5umjdvnjp27KjatWurT58+KlWqlP7880/t2rXLsjSBvdzc3PTuu+8qIiJC1apVU58+fXTXXXfpn3/+0dq1a+Xr66tvv/1W0tWXDK9cuVKNGzfWs88+axn0ql69uuLj42063xtvvKG1a9eqYcOG6t+/v6pWrarTp09r69atWrVqleWGUZs2bRQcHKzGjRsrKChIe/bs0VtvvaUOHTrIx8dHZ8+e1d13361HHnlEtWrVUtGiRbVq1Spt3rxZU6dOveH5O3bsqJYtW+rll1/WoUOHVKtWLa1cuVJff/21hg4dmuU9NwAAwPF8fX0t7/VNTU3VXXfdpZUrVyohIcHuNmvXrq3HH39cc+fO1blz53T//fdr9erV2c7ynjBhgmJiYtSkSRM999xzKlSokN5++22lpKRo8uTJt3NpDmFrZr2ZJ598Up999pmeeeYZrV27Vo0bN1Z6err+/PNPffbZZ/rpp59Uv359jR8/XrGxserQoYNCQ0N14sQJzZ07V3fffbeaNGki6erEM39/f82fP18+Pj7y9vZWw4YNC9ySWAAA4OYmTJig+Ph4rV69Wj4+PqpZs6bGjBmjV155RY888ojat2+viRMnqlOnTmrcuLH69OmjM2fOWO4NZTcJvEKFCmrSpImeffZZ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YzdHvWclISFBiYmJCg8Pt5T5+fmpYcOG2rBhgyTeswIAAJzP0VkqN93sBlOm7G4wxcbGKjU11VLHnhtMAAAA2XFmlrpw4YLi4+MVHx8v6eq9qPj4eB05ckQXLlzQiBEjtHHjRh06dEirV69Wp06dVKFCBbVt21bS1aft27Vrp/79++v333/Xb7/9pkGDBql79+4KCQmRJPXo0UMeHh7q16+fdu3apU8//VQzZ87MspLCrThtsM7NzU3u7u433QoVyhcP/gEAUKCZTCaHbo56kW/me1Kyewnvte9QyasX+RY0ZCkAAPKGo7NUThSkG0wFDVkKAIC84cwstWXLFtWpU8eytPXw4cNVp04djRkzRu7u7tq+fbseeugh3XPPPerXr5/q1aunX375xWoVhCVLlqhy5cpq3bq12rdvryZNmmjBggWW/X5+flq5cqUSEhJUr149vfDCCxozZkyOV3tyWupYtmzZDfdt2LBBs2bNUkZGRh72CAAA2MKR71mB/chSAAC4vi1btqhly5aWz5kZLDIyUvPmzdP27du1ePFinT17ViEhIWrTpo1ee+21LDeYBg0apNatW8vNzU1du3bVrFmzLPszbzANHDhQ9erVU4kSJey6wVTQkKUAAHB9LVq0kGEYN9z/008/3bKNgIAALV269KZ1atasqV9++SXH/buW0wbrMt9Nc629e/fqpZde0rfffquePXtq/PjxTugZAACuxdHLLTnqPSuZ70A5fvy4ZammzM+1a9e21OE9K9kjSwEAkDdye+nKmylIN5gKGrIUAAB5w5lZqiDJF++sO3r0qPr3768aNWooLS1N8fHxWrx4sUJDQ53dNQAACjxnLjdwM2FhYQoODtbq1astZUlJSdq0aZMaNWokifes2IosBQBA7smvWQqOQ5YCACD3kKVs49TBunPnzmnUqFGqUKGCdu3apdWrV+vbb79V9erVndktAABciimfvsjXZDJp6NChmjBhgr755hvt2LFDvXr1UkhIiDp37iyJ96zcClkKAIDc58wshdxFlgIAIPeRpWzjtGUwJ0+erEmTJik4OFgff/xxtssPAACAgu1m71lZtGiRRo4cqeTkZA0YMEBnz55VkyZNtGLFCnl6elqO4T0r2SNLAQAA2I8sBQAA8hOTcbPFz3ORm5ubvLy8FB4eLnd39xvW++qrr3LctledQbfTNQDZeG78YGd3AXA5UztWypPzNJoU69D2Noxq5tD2YB+yFFCwnNn8lrO7ALgczzyafkyWck1kKaBgIUsBjkeWyl+c9mRdr169XHp9UQAA8gv+3LomshQAAHmDP7euiSwFAEDe4M+tbZw2WLdo0SJnnRoAAKDAI0sBAADYjywFAADyE6cN1gEAgLzBjGEAAAD7kaUAAADsR5ayDYN1AAC4ODIRAACA/chSAAAA9iNL2cbN2R0AAAAAAAAAAAAA7lQ8WQcAgItjuQEAAAD7kaUAAADsR5ayDYN1AAC4OEIRAACA/chSAAAA9iNL2YZlMAEAAAAAAAAAAAAn4ck6AABcHBOYAAAA7EeWAgAAsB9ZyjYM1gEA4OJYbgAAAMB+ZCkAAAD7kaVswzKYAAAAAAAAAAAAgJPwZB0AAC6OCUwAAAD2I0sBAADYjyxlGwbrAABwcSw3AAAAYD+yFAAAgP3IUrZhGUwAAAAAAAAAAADASXiyDgAAF8cEJgAAAPuRpQAAAOxHlrINg3UAALg4N1IRAACA3chSAAAA9iNL2YZlMAEAAAAAAAAAAAAn4ck6AABcHBOYAAAA7EeWAgAAsB9ZyjYM1gEA4OJMpCIAAAC7kaUAAADsR5ayDctgAgAAAAAAAAAAAE7Ck3UAALg4NyYwAQAA2I0sBQAAYD+ylG0YrAMAwMWx3AAAAID9yFIAAAD2I0vZhmUwAQAAAAAAAAAAACfhyToAAFwcE5gAAADsR5YCAACwH1nKNgzWAQDg4kwiFQEAANiLLAUAAGA/spRtWAYTAAAAAAAAAAAAcBKerAMAwMW5MYEJAADAbmQpAAAA+5GlbMNgHQAALs7E4uAAAAB2I0sBAADYjyxlG5bBBAAAAAAAAAAAAJyEwToAAFycyeTYDQAA4E5ClgIAALCfM7NUbGysOnbsqJCQEJlMJi1fvtyyLzU1VaNGjVKNGjXk7e2tkJAQ9erVS0ePHrVqo2zZsjKZTFbbG2+8YVVn+/btatq0qTw9PVW6dGlNnjw5x98Tg3UAALg4N5PJoRsAAMCdxJlZqiDdYAIAAMiOM7NUcnKyatWqpTlz5mTZd/HiRW3dulWvvvqqtm7dqq+++kp79+7VQw89lKXu+PHjdezYMcs2ePBgy76kpCS1adNGoaGhiouL05QpUxQVFaUFCxbkqK+8sw4AAAAAACAfyrzB1LdvX3Xp0sVq37U3mGrVqqUzZ87o+eef10MPPaQtW7ZY1R0/frz69+9v+ezj42P5d+YNpvDwcM2fP187duxQ37595e/vrwEDBuTuBQIAAORQSkqKUlJSrMrMZrPMZnOWuhEREYqIiMi2HT8/P8XExFiVvfXWW2rQoIGOHDmiMmXKWMp9fHwUHBycbTtLlizRlStX9P7778vDw0PVqlVTfHy8pk2blqMsxZN1AAC4OJZuAgAAsJ+js1RKSoqSkpKstutvOGWKiIjQhAkT9PDDD2fZl3mDqVu3bqpUqZLuu+8+vfXWW4qLi9ORI0es6mbeYMrcvL29LfuuvcFUrVo1de/eXUOGDNG0adMc+0UCAIA7kqOzVHR0tPz8/Ky26Ohoh/T13LlzMplM8vf3typ/4403VLx4cdWpU0dTpkxRWlqaZd+GDRvUrFkzeXh4WMratm2rvXv36syZMzafm8E6AABc3PXLHt3uBgAAcCdxdJZy1RtMAAAA2XF0lho9erTOnTtntY0ePfq2+3n58mWNGjVKjz/+uHx9fS3lQ4YM0SeffKK1a9fq6aef1sSJEzVy5EjL/sTERAUFBVm1lfk5MTHR5vOzDCYAAAAAAEAeGT16tIYPH25Vlt2yTTl1sxtMdevWVUBAgNavX6/Ro0fr2LFjlifnEhMTFRYWZtXWtTeYihUrdtt9AwAAcJQbLXl5O1JTU9WtWzcZhqF58+ZZ7bs2t9WsWVMeHh56+umnFR0d7dB+MFgHAICL42E4AAAA+zk6S7nqDSYAAIDs5Pf7Upk56vDhw1qzZo3VpKfsNGzYUGlpaTp06JAqVaqk4OBgHT9+3KpO5ucbvecuOyyDCQCAi3MzmRy6AQAA3Enye5a69gZTTExMjm4wSXLYDSYAAIDs5OcslZmj9u3bp1WrVql48eK3PCY+Pl5ubm4KDAyUJDVq1EixsbFKTU211ImJiVGlSpVytEIBg3UAAAAAAAAFUH66wQQAAJDfXLhwQfHx8YqPj5ckJSQkKD4+XkeOHFFqaqoeeeQRbdmyRUuWLFF6eroSExOVmJioK1euSLr6bt8ZM2bojz/+0MGDB7VkyRINGzZMTzzxhCUn9ejRQx4eHurXr5927dqlTz/9VDNnzsyy7PmtsAwmAAAujmfhAAAA7OfMLHXhwgXt37/f8jnzBlNAQIBKlSqlRx55RFu3btV3331nucEkSQEBAfLw8NCGDRu0adMmtWzZUj4+PtqwYUO2N5jGjRunfv36adSoUdq5c6dmzpyp6dOnO+WaAQCAa3FmltqyZYtatmxp+Zw5gBYZGamoqCh98803kqTatWtbHbd27Vq1aNFCZrNZn3zyiaKiopSSkqKwsDANGzbMaiDOz89PK1eu1MCBA1WvXj2VKFFCY8aM0YABA3LUVwbrAABwcSaWrgQAALCbM7NUQbrBBAAAkB1nZqkWLVrIMIwb7r/ZPkmqW7euNm7ceMvz1KxZU7/88kuO+3ctBusAAAAAAADyoYJ0gwkAAAD2Y7AOAAAX58aDdQAAAHYjSwEAANiPLGUbBusAAHBxLIMJAABgP7IUAACA/chStnFzdgcAAIBrSk9P16uvvqqwsDB5eXmpfPnyeu2116yWazIMQ2PGjFGpUqXk5eWl8PBw7du3z6qd06dPq2fPnvL19ZW/v7/69eunCxcu5PXlAAAAAAAAALnCpifrMl9YbIuHHnrI7s4AAADHc9YEpkmTJmnevHlavHixqlWrpi1btqhPnz7y8/PTkCFDJEmTJ0/WrFmztHjxYoWFhenVV19V27ZttXv3bnl6ekqSevbsqWPHjikmJkapqanq06ePBgwYoKVLlzrnwuxAlgIAoOBiMrjzkaUAACi4yFK2sWmwrnPnzjY1ZjKZlJ6efjv9AQAADuas5QbWr1+vTp06qUOHDpKksmXL6uOPP9bvv/8u6epTdTNmzNArr7yiTp06SZI++OADBQUFafny5erevbv27NmjFStWaPPmzapfv74kafbs2Wrfvr3efPNNhYSEOOXacoosBQBAwcXSTc5HlgIAoOAiS9nGpmUwMzIybNoIRAAAuL6UlBQlJSVZbSkpKVnq3X///Vq9erX++usvSdIff/yhX3/9VREREZKkhIQEJSYmKjw83HKMn5+fGjZsqA0bNkiSNmzYIH9/f8tAnSSFh4fLzc1NmzZtys3LdCiyFAAAgP3IUgAAwNXxzjoAAFycm8mxW3R0tPz8/Ky26OjoLOd96aWX1L17d1WuXFmFCxdWnTp1NHToUPXs2VOSlJiYKEkKCgqyOi4oKMiyLzExUYGBgVb7CxUqpICAAEsdAACA3OToLAUAAHAnIUvZxqZlMK+XnJysn3/+WUeOHNGVK1es9mW+gwYAAOQPjl5uYPTo0Ro+fLhVmdlszlLvs88+05IlS7R06VJVq1ZN8fHxGjp0qEJCQhQZGenQPhU0ZCkAAAoOlm7Kf8hSAAAUHGQp2+R4sG7btm1q3769Ll68qOTkZAUEBOjff/9VkSJFFBgYSCgCAMDFmc3mbAfnrjdixAjL03WSVKNGDR0+fFjR0dGKjIxUcHCwJOn48eMqVaqU5bjjx4+rdu3akqTg4GCdOHHCqt20tDSdPn3acnxBQ5YCAACwH1kKAAC4ohwvgzls2DB17NhRZ86ckZeXlzZu3KjDhw+rXr16evPNN3OjjwAA4DaYHLzZ6uLFi3Jzs44a7u7uysjIkCSFhYUpODhYq1evtuxPSkrSpk2b1KhRI0lSo0aNdPbsWcXFxVnqrFmzRhkZGWrYsGEOepN/kKUAAChYnJWlkD2yFAAABQtZyjY5frIuPj5eb7/9ttzc3OTu7q6UlBSVK1dOkydPVmRkpLp06ZIb/QQAAHZyc9JyAx07dtTrr7+uMmXKqFq1atq2bZumTZumvn37Srq6DMLQoUM1YcIEVaxYUWFhYXr11VcVEhKizp07S5KqVKmidu3aqX///po/f75SU1M1aNAgde/eXSEhIU65rttFlgIAoGBxVpZC9shSAAAULGQp2+R4sK5w4cKWWfKBgYE6cuSIqlSpIj8/P/33v/91eAcBAEDBNHv2bL366qt67rnndOLECYWEhOjpp5/WmDFjLHVGjhyp5ORkDRgwQGfPnlWTJk20YsUKeXp6WuosWbJEgwYNUuvWreXm5qauXbtq1qxZzrgkhyBLAQAA2I8sBQAAXFGOB+vq1KmjzZs3q2LFimrevLnGjBmjf//9Vx9++KGqV6+eG30EAAC3wVkTmHx8fDRjxgzNmDHjhnVMJpPGjx+v8ePH37BOQECAli5dmgs9dA6yFAAABQuTwfMXshQAAAULWco2OX5n3cSJE1WqVClJ0uuvv65ixYrp2Wef1cmTJ7VgwQKHdxAAANwek8nk0A23hywFAEDBQpbKX8hSAAAULGQp2+T4ybr69etb/h0YGKgVK1Y4tEMAAACujCwFAABgP7IUAABwRTkerAMAAAWLC086AgAAyHVkKQAAAPuRpWyT48G6sLCwmz5qePDgwdvqEAAAcCw3UlG+QpYCAKBgIUvlL2QpAAAKFrKUbXI8WDd06FCrz6mpqdq2bZtWrFihESNGOKpfAAAALoksBQAAYD+yFAAAcEU5Hqx7/vnnsy2fM2eOtmzZctsdAgAAjsUEpvyFLAUAQMFClspfyFIAABQsZCnbuDmqoYiICH355ZeOag4AADiIyWRy6IbcQZYCACB/IksVDGQpAADyJ7KUbRw2WPfFF18oICDAUc0BAADcUchSAAAA9iNLAQCAgizHy2DWqVPHavTSMAwlJibq5MmTmjt3rkM7Z68zm99ydhcAl5OckubsLgCwk8Nm5sAhCkKWOvLLDGd3AQCAfIMslb8UhCz1z68znd0FAADyDbKUbXI8WNepUyerUOTm5qaSJUuqRYsWqly5skM7BwAAbp8rLxFQEJGlAAAoWMhS+QtZCgCAgoUsZZscD9ZFRUXlQjcAAADuDGQpAAAA+5GlAACAK8rxE4ju7u46ceJElvJTp07J3d3dIZ0CAACO42Zy7IbbQ5YCAKBgIUvlL2QpAAAKFrKUbXL8ZJ1hGNmWp6SkyMPD47Y7BAAAHMuVg0xBRJYCAKBgIUvlL2QpAAAKFrKUbWwerJs1a5akq+uLvvvuuypatKhlX3p6umJjY1kbHAAA4AbIUgAAAPYjSwEAAFdm82Dd9OnTJV2dwTR//nyrpQU8PDxUtmxZzZ8/3/E9BAAAt4UX+eYPZCkAAAomslT+QJYCAKBgIkvZxubBuoSEBElSy5Yt9dVXX6lYsWK51ikAAOA4LDeQP5ClAAAomMhS+QNZCgCAgoksZZscv7Nu7dq1udEPAACAOwJZCgAAwH5kKQAA4IrccnpA165dNWnSpCzlkydP1qOPPuqQTgEAAMcxmRy74faQpQAAKFjIUvkLWQoAgIKFLGWbHA/WxcbGqn379lnKIyIiFBsb65BOAQAAx3EzmRy64faQpQAAKFjIUvkLWQoAgIKFLGWbHA/WXbhwQR4eHlnKCxcurKSkJId0CgAAwFWRpQAAAOxHlgIAAK4ox4N1NWrU0Keffpql/JNPPlHVqlUd0ikAAOA4bg7ecHvIUgAAFCzOzFKxsbHq2LGjQkJCZDKZtHz5cqv9hmFozJgxKlWqlLy8vBQeHq59+/ZZ1Tl9+rR69uwpX19f+fv7q1+/frpw4YJVne3bt6tp06by9PRU6dKlNXny5Bz2NO+QpQAAKFjIUrYplNMDXn31VXXp0kUHDhxQq1atJEmrV6/W0qVL9cUXX+S4AwAAIHe58AoBBRJZCgCAgsWZWSo5OVm1atVS37591aVLlyz7J0+erFmzZmnx4sUKCwvTq6++qrZt22r37t3y9PSUJPXs2VPHjh1TTEyMUlNT1adPHw0YMEBLly6VJCUlJalNmzYKDw/X/PnztWPHDvXt21f+/v4aMGBAnl6vLchSAAAULGQp25gMwzByeoHff/+9Jk6cqPj4eHl5ealWrVoaO3asAgICVL169Zw253CX05zdA8D1JKfwgwU4WnHvHM+ZscvLP/7l0PZej7jHoe3difJ7ljp5gd/5gKP5eObN73zgTpJXP1aOzlJjWoUqJSXFqsxsNstsNt/0OJPJpGXLlqlz586Srs4EDwkJ0QsvvKAXX3xRknTu3DkFBQVp0aJF6t69u/bs2aOqVatq8+bNql+/viRpxYoVat++vf7++2+FhIRo3rx5evnll5WYmGhZXvKll17S8uXL9eeffzr02h0lv2ep08npzu4C4HKKmN2d3QXA5ZCl8leWsms1qw4dOui3335TcnKyDh48qG7duunFF19UrVq17GkOAADkIl7km/+QpQAAKDgcnaWio6Pl5+dntUVHR+e4XwkJCUpMTFR4eLilzM/PTw0bNtSGDRskSRs2bJC/v7/l5pIkhYeHy83NTZs2bbLUadasmdV74Nq2bau9e/fqzJkz9n5tuYosBQBAwUGWsvF7yvEV/L/Y2FhFRkYqJCREU6dOVatWrbRx40Z7mwMAALnEZHLsBscgSwEAUDA4OkuNHj1a586ds9pGjx6d434lJiZKkoKCgqzKg4KCLPsSExMVGBhotb9QoUIKCAiwqpNdG9eeIz8iSwEAUDCQpWyTowcdExMTtWjRIr333ntKSkpSt27dlJKSouXLl/MSXwAAgFsgSwEAAFuWaUL2yFIAAMBVs5TNT9Z17NhRlSpV0vbt2zVjxgwdPXpUs2fPzs2+AQAAB3AzOXaDfchSAAAUTPk1SwUHB0uSjh8/blV+/Phxy77g4GCdOHHCan9aWppOnz5tVSe7Nq49R35AlgIAoGAiS9nG5sG6H3/8Uf369dO4cePUoUMHubvzUk8AAAoC3lmXP5ClAAAomPJrlgoLC1NwcLBWr15tKUtKStKmTZvUqFEjSVKjRo109uxZxcXFWeqsWbNGGRkZatiwoaVObGysUlNTLXViYmJUqVIlFStWzGH9vV1kKQAACiaylG1sHqz79ddfdf78edWrV08NGzbUW2+9pX///dfmEwEAANzJyFIAACCnLly4oPj4eMXHx0uSEhISFB8fryNHjshkMmno0KGaMGGCvvnmG+3YsUO9evVSSEiIOnfuLEmqUqWK2rVrp/79++v333/Xb7/9pkGDBql79+4KCQmRJPXo0UMeHh7q16+fdu3apU8//VQzZ87U8OHDnXTV2SNLAQCAnCpIWcrmwbr77rtP77zzjo4dO6ann35an3zyiUJCQpSRkaGYmBidP38+RycGAAB5w9Ev8oV9yFIAABRMzsxSW7ZsUZ06dVSnTh1J0vDhw1WnTh2NGTNGkjRy5EgNHjxYAwYM0L333qsLFy5oxYoV8vT0tLSxZMkSVa5cWa1bt1b79u3VpEkTLViwwLLfz89PK1euVEJCgurVq6cXXnhBY8aM0YABA27/y3MgshQAAAUTWcrG78kwDCNnl/c/e/fu1XvvvacPP/xQZ8+e1QMPPKBvvvnG3uYc5nKas3sAuJ7kFH6wAEcr7l0oT87z+ur9Dm3v5dYVHNrenSy/ZqmTF/idDziaj2fe/M4H7iR59WNFlsq/8muWOp2c7uwuAC6niJmlbwFHI0vlLzY/WZedSpUqafLkyfr777/18ccfO6pPAAAAdwSyFAAAgP3IUgAAwFU4ZOzU3d1dnTt3tqzjCQAA8g+TWLsyvyNLAQCQf5Gl8j+yFAAA+RdZyjasxQIAgItzIxMBAADYjSwFAABgP7KUbW5rGUwAAAAAAAAAAAAA9uPJOgAAXBwzmAAAAOxHlgIAALAfWco2DNYBAODiTCZSEQAAgL3IUgAAAPYjS9mGZTABAAAAAAAAAAAAJ+HJOgAAXBzLDQAAANiPLAUAAGA/spRtGKwDAMDFsdoAAACA/chSAAAA9iNL2YZlMAEAAAAAAAAAAAAn4ck6AABcnBtTmAAAAOxGlgIAALAfWco2DNYBAODiWBscAADAfmQpAAAA+5GlbMMymAAAINf8888/euKJJ1S8eHF5eXmpRo0a2rJli2W/YRgaM2aMSpUqJS8vL4WHh2vfvn1WbZw+fVo9e/aUr6+v/P391a9fP124cCGvLwUAAAAAAADIFQzWAQDg4kwmx262OnPmjBo3bqzChQvrxx9/1O7duzV16lQVK1bMUmfy5MmaNWuW5s+fr02bNsnb21tt27bV5cuXLXV69uypXbt2KSYmRt99951iY2M1YMAAR35FAAAAN+SsLAUAAOAKyFK2YRlMAABcnJscm2RSUlKUkpJiVWY2m2U2m63KJk2apNKlS2vhwoWWsrCwMMu/DcPQjBkz9Morr6hTp06SpA8++EBBQUFavny5unfvrj179mjFihXavHmz6tevL0maPXu22rdvrzfffFMhISEOvTYAAIDrOTpLAQAA3EnIUrbhyToAAJAj0dHR8vPzs9qio6Oz1Pvmm29Uv359PfroowoMDFSdOnX0zjvvWPYnJCQoMTFR4eHhljI/Pz81bNhQGzZskCRt2LBB/v7+loE6SQoPD5ebm5s2bdqUi1cJAAAAAAAA5A2erAMAwMU5eomA0aNHa/jw4VZl1z9VJ0kHDx7UvHnzNHz4cP3nP//R5s2bNWTIEHl4eCgyMlKJiYmSpKCgIKvjgoKCLPsSExMVGBhotb9QoUIKCAiw1AEAAMhNrrzcEgAAQG4jS9mGwToAAFycm4NDUXZLXmYnIyND9evX18SJEyVJderU0c6dOzV//nxFRkY6tlMAAAC5xNFZCgAA4E5ClrINy2ACAIBcUapUKVWtWtWqrEqVKjpy5IgkKTg4WJJ0/PhxqzrHjx+37AsODtaJEyes9qelpen06dOWOgAAAAAAAEBBxmAdAAAuzs1kcuhmq8aNG2vv3r1WZX/99ZdCQ0MlSWFhYQoODtbq1ast+5OSkrRp0yY1atRIktSoUSOdPXtWcXFxljpr1qxRRkaGGjZseDtfCwAAgE2claUAAABcAVnKNiyDCQCAi3NWjhk2bJjuv/9+TZw4Ud26ddPvv/+uBQsWaMGCBf/fL5OGDh2qCRMmqGLFigoLC9Orr76qkJAQde7cWdLVJ/HatWun/v37a/78+UpNTdWgQYPUvXt3hYSEOOfCAADAHcWF7wkBAADkOrKUbRisAwAAueLee+/VsmXLNHr0aI0fP15hYWGaMWOGevbsaakzcuRIJScna8CAATp79qyaNGmiFStWyNPT01JnyZIlGjRokFq3bi03Nzd17dpVs2bNcsYlAQAAAAAAAA5nMgzDcHYnHO1ymrN7ALie5BR+sABHK+6dN3Nm3vv9iEPb69egjEPbQ/5z8gK/8wFH8/FkniTgaHn1Y0WWQk6dTk53dhcAl1PE7O7sLgAuhyyVv/B/jAAAuDiWGwAAALAfWQoAAMB+ZCnbuDm7AwAAAAAAAAAAAMCdiifrAABwcczMAQAAsB9ZCgAAwH5kKdswWAcAgIszsd4AAACA3chSAAAA9iNL2YZBTQAAAAAAAAAAAMBJeLIOAAAXx/wlAAAA+5GlAAAA7EeWsg2DdQAAuDg3lhsAAACwG1kKAADAfmQp27AMJgAAAAAAAAAAAOAkPFkHAICLY/4SAACA/chSAAAA9iNL2YbBOgAAXByrDQAAANiPLAUAAGA/spRtWAYTAAAAAAAgnylbtqxMJlOWbeDAgZKkFi1aZNn3zDPPWLVx5MgRdejQQUWKFFFgYKBGjBihtLQ0Z1wOAABAnipoWYon6wAAcHEmpjABAADYzVlZavPmzUpPT7d83rlzpx544AE9+uijlrL+/ftr/Pjxls9FihSx/Ds9PV0dOnRQcHCw1q9fr2PHjqlXr14qXLiwJk6cmDcXAQAA7nhkKdswWAcAgIvjMXoAAAD7OTpLpaSkKCUlxarMbDbLbDZblZUsWdLq8xtvvKHy5curefPmlrIiRYooODg42/OsXLlSu3fv1qpVqxQUFKTatWvrtdde06hRoxQVFSUPDw8HXREAAMCNOeu+VEHLUty/AwAAAAAAyCPR0dHy8/Oz2qKjo296zJUrV/TRRx+pb9++VrPTlyxZohIlSqh69eoaPXq0Ll68aNm3YcMG1ahRQ0FBQZaytm3bKikpSbt27XL8hQEAAOSBlJQUJSUlWW3XT4S6XkHIUjxZBwCAi2MZTAAAAPs5OkuNHj1aw4cPtyq7/qm66y1fvlxnz55V7969LWU9evRQaGioQkJCtH37do0aNUp79+7VV199JUlKTEy0urkkyfI5MTHRAVcCAABwa47OUtHR0Ro3bpxV2dixYxUVFXXDYwpClmKwDgAAF8dQHQAAgP0cnaWyW/LyVt577z1FREQoJCTEUjZgwADLv2vUqKFSpUqpdevWOnDggMqXL++w/gIAANwOR2cpeyY+FYQsxTKYAAAAAAAA+dThw4e1atUqPfXUUzet17BhQ0nS/v37JUnBwcE6fvy4VZ3Mzzd6NwsAAEB+Zzab5evra7XdbLCuoGQpBusAAHBxJpPJoRsAAMCdxNlZauHChQoMDFSHDh1uWi8+Pl6SVKpUKUlSo0aNtGPHDp04ccJSJyYmRr6+vqpatWqO+wEAAGAPspRtWAYTAAAXx8wcAAAA+zkzS2VkZGjhwoWKjIxUoUL/u4Vz4MABLV26VO3bt1fx4sW1fft2DRs2TM2aNVPNmjUlSW3atFHVqlX15JNPavLkyUpMTNQrr7yigQMH5ngZTgAAAHuRpWzDYB0AAAAAAEA+tGrVKh05ckR9+/a1Kvfw8NCqVas0Y8YMJScnq3Tp0uratateeeUVSx13d3d99913evbZZ9WoUSN5e3srMjJS48ePz+vLAAAAcIqClKVMhmEYudKyE11Oc3YPANeTnMIPFuBoxb3zZs7Msu2JDm3v4Zq848TVnbzA73zA0Xw8mScJOFpe/ViRpZBTp5PTnd0FwOUUMbs7uwuAyyFL5S/8HyMAAC6Ot8wBAADYjywFAABgP7KUbXiNDQAAAAAAAAAAAOAkPFkHAICLMzGFCQAAwG5kKQAAAPuRpWzDYB0AAC7OjQUHAAAA7EaWAgAAsB9ZyjYsgwkAAAAAAAAAAAA4CU/WAQDg4lhuAAAAwH5kKQAAAPuRpWzDYB0AAC7OxHIDAAAAdiNLAQAA2I8sZRuWwQQAAAAAAAAAAACchCfrAABwcSw3AAAAYD+yFAAAgP3IUrZhsA4AABfnxnIDAAAAdiNLAQAA2I8sZRuWwQQAAAAAAAAAAACchCfrAABwcSw3AAAAYD+yFAAAgP3IUrZhsA4AABdHKAIAALAfWQoAAMB+ZCnbsAwmAAAAAAAAAAAA4CROe7KuS5cuNtf96quvcrEnAAC4NhMv8nVJZCkAAPIGWco1kaUAAMgbZCnbOG2wzs/Pz/JvwzC0bNky+fn5qX79+pKkuLg4nT17NkfhCQAAZOVGJnJJZCkAAPIGWco1kaUAAMgbZCnbOG2wbuHChZZ/jxo1St26ddP8+fPl7u4uSUpPT9dzzz0nX19fZ3URAAAg3yJLAQAA2I8sBQAA8hOTYRiGsztRsmRJ/frrr6pUqZJV+d69e3X//ffr1KlTOWrvcpojewdAkpJT+MECHK24d97MmVnzZ87+jt5Kq8rFHdoebp+js9TJC/zOBxzNx9Np8yQBl5VXP1ZkKdfn6Cx1Ojndkd0DIKmI2d3ZXQBcDlkqf3FzdgckKS0tTX/++WeW8j///FMZGRlO6BEAAK7DZHLshvyHLAUAQO4hS7k+shQAALmHLGWbfDG9s0+fPurXr58OHDigBg0aSJI2bdqkN954Q3369HFy7wAAAPI3shQAAID9yFIAAMDZ8sVg3Ztvvqng4GBNnTpVx44dkySVKlVKI0aM0AsvvODk3gEAULCZ5MLTjiCJLAUAQG4iS7k+shQAALmHLGWbfPHOumslJSVJ0m29wJd31gGOxzvrAMfLq3fWxf512qHtNbsnwKHtwbEckaV4Zx3geLyzDnC8vPqxIkvdWRyRpXhnHeB4vLMOcDyyVP6SL95Zdy1fX9/bCkQAACB/euONN2QymTR06FBL2eXLlzVw4EAVL15cRYsWVdeuXXX8+HGr444cOaIOHTqoSJEiCgwM1IgRI5SWxmDSjZClAAAA7EeWAgAAzpAvpneGhYXJdJM3Ax48eDAPewMAgGvJD8sNbN68WW+//bZq1qxpVT5s2DB9//33+vzzz+Xn56dBgwapS5cu+u233yRJ6enp6tChg4KDg7V+/XodO3ZMvXr1UuHChTVx4kRnXEq+RJYCACD35IcshdxFlgIAIPeQpWyTLwbrrp1hL0mpqanatm2bVqxYoREjRjinU3CI9955W6tjVioh4aDMnp6qXbuOhg5/UWXDylnq/PfIEU19c5Lit8bpypUratykqV76z6sqXqKEE3sO5B/b4rZo6Qfva++e3fr335OKnjpLzVu2tuxftzpGy778THv37FLSuXNa9PEXuqdSFas2UlJSNHvaZK1a+aNSr1xRw0aN9eLoVxVQnJ+zO8FN7jvYJSUlRSkpKVZlZrNZZrM52/oXLlxQz5499c4772jChAmW8nPnzum9997T0qVL1apVK0nSwoULVaVKFW3cuFH33XefVq5cqd27d2vVqlUKCgpS7dq19dprr2nUqFGKioqSh4eHYy+ugCJLuY74rf/7nX/q35Oa+OYsNbvmd/57b8/R6p9+1InjiSpUuLAqVamqAc89r2o1/jcQfuTwIc2d+aZ2xG9Talqqyle4R/2fHay69zZ0xiUBBULcls1a9P572rN7p06ePKnps+aoVetwZ3cL+YSjsxTyH7KU69gWt0VLPnhfe/fs0r//ntQbU2epecurv8/TUlP19txZWv9brI7+/beKFi2q+g0b6bkhw1WyZKBVO7/98rPef2eu9u/7S2YPs+rUq69J095yxiUB+dKtstO8ObO14sfvlZiYqMKFC6tq1Woa9Pww1axZy4m9hrOQpWyTLwbrnn/++WzL58yZoy1btuRxb+BIWzb/rsce76lqNWooPS1ds2dO0zP9++mrb75XkSJFdPHiRT0zoK/uqVRZ77y/WJI0Z/ZMDR74jD76+DO5ueW7lVqBPHf58iVVuKeSHuzURaNfzPr78tKlS6pVu45aP9BWb7w2Nts2Zk2dpPW//qwJk6apaFEfTZ30uka/+LzeXrgkt7sPFxQdHa1x48ZZlY0d+3/t3Xd8VFX+//H3JCEhhBRTSIJAaEIooepCVqWJBCxL29/awAQRNQRWCM0oQoSFAPulWALoVyCgKNKVoggosFJE46KAEAUFBBJAMCBg+vz+4MvIUGQYb3Inw+vpYx4P5t4zd84M3uE98znn3NFKTU29avukpCTdf//96tixo12xLjMzU4WFherY8fdAHx0drRo1amjr1q1q3bq1tm7dqpiYGIWHh9vaxMXFKTExUbt371bz5s2NfXHlFFnKffz224XP/Pv/1kMvDLvy77V6jSgNHvGCqt5aTfn5+Vo4f56Sk/ppwfsf6pZbLqzbP3xQf1WvHqWXX58tH5+KWvjOPA0flKT33v9QIaFhZf2SgHLht9/Oq379+urWo6eSnx1gdncAlDGylPvIyzuv22zfn/952b48Ze39Vn2efEa31YvWr2fOaOr/jNfwQUmaM3+Rrd2n6z9W2thRembAIN1+R2sVFxdp/77vy/qlAC7tetkpKqqmUl4YpWrVqisvP09vz8tQYr8ntOLDtQoOds/rjQF/lksU666lS5cuSklJ0Zw5c8zuCpw0441ZdvfHjJug9nfHas+3u9Xy9ju0479f6eiRI3pv8XJVrlxZkjR2/ETdHXuHtn++Ta1j/2pGtwGXEnvn3Yq98+5r7u/ywN8kSdlHj1x1/9lff9WK5UuUOn6Sbv9La0nSC6n/0qM9H9Sub75WY0Y1uT2jBzClpKQoOTnZbtu1ZtUtWLBAX331lb744osr9uXk5Mjb21tBQUF228PDw5WTk2Nrc2mh7uL+i/vwx8hS5c/1PvM7dXnA7v7A5OFa+f4S7f/+O93+l9bK/eUXHT50UCmjxqrubfUlSYkDk7Vs0QL9sH8fxTrgGu66u63uurut2d2Ai2Iw+M2LLFX+xN7ZRrF3trnqvsr+/nplhv3vVENGjFTf3g8pJ/uoIiKrqqioSFP/naYBg4bpb9162trVql23VPsNlDfXy073PfCg3f2hw1O0bMliff9dllq1ji3t7sHFkKUc49LFusWLF1NpdzNnf/1VkhQQGChJKigokMVisVvGzMfHRx4eHvrvV5kU6wAD7N2zW0VFRbqj1e9hqGat2gqPiNSub3ZQrLsJeBi83sAfLXl5qZ9++knPPvus1q5dq4oVKxraBziGLOXeCgsL9P7SRapc2d9WmAsMClKNqFr6aOX7qhfdQBUqeGv5koW6JThE9Rs0NLnHAFA+GZ2lUH6Qpdzf2bO/ymKxyN8/QJKUtfdbnTh+TB4Wix5/pIdOnfxZt9WL1oBBw1Sn7m0m9xYonwoLCrRk0Xvy9/dXvfr1ze4OTECWcoxLFOuaN29udyFfq9WqnJwcnThxQtOnT//Dx17tujlWT8d+RETZKikp0aSJ49WseQvddls9SVKTps3k6+uraZP/rYGDkmW1WvXy1MkqLi7WiRMnTO4x4B5OnfxZFSpUsH35uCg4JEQnT/5sUq9wM8jMzNTx48fVokUL27bi4mJt2rRJr732mtasWaOCggLl5ubaza47duyYIiIiJEkRERHavn273XGPHTtm24cLjM5S+YWeZCkXtnnTBqU+P1R5eXkKCQ3T1On/q6BbbpEkWSwWTZvxplKG/FOd7v6LPDw8FHRLsCa/+roCAgJN7jkAAK7J8CxV5EWWKgfy8/M1/eUpurfzffL7v9Wejh45LOnCdYL/OWSEIiNv1TtvZyjpqXi9t2y1AgODTOwxUL5s3PCpRgxNVl7ebwoNC9PM/51tW7ofwJVcoljXrVs3u/seHh4KCwtTu3btFB0d/YePvdp1c154cbRGjko1uJf4s8b/6yXt//57Zbz1jm1bcHCw/j3lZY0bm6p35r8lDw8Pdb7vfjVo2EgeHlTcAcAIZn2a3nPPPdq5c6fdtj59+ig6OlojRoxQ9erVVaFCBa1fv149e15YYiYrK0uHDh1SbOyFmaCxsbEaN26cjh8/ripVLlz0fe3atQoICFDDhswSusjoLDU05UUNf36U0d2EQVrc8RfNeXeJcnNztWLZYo16bojemPuubgkOkdVq1ZSJ/9ItwcFKf3OefHwqasXyxRoxOEn/O+89hYaxDCYA3Ci+mbo/o7PU8JQXNeKFq19PHK6hqLBQI0ckyyqrhqf8/ndVUlIiSYrv+7Ta39NJkjQydZy6dm6vT9auUfe/P2RKf4Hy6I6/tNLCJcuVm/uLlixeqGFDBuntdxcpJCTE7K6hjJGlHOMSxbrRo50PMFe7bo7Vk9FLrmb8v8Zo08YNmj33bYVfNhPir3fepVUfrdMvv5ySp6eXAgIC1KHNnarW5T6Tegu4l+CQUBUWFurXX8/Yza47dfKkQkJCTewZyoxJqcjf31+NGze22+bn56eQkBDb9r59+yo5OVnBwcEKCAjQwIEDFRsbq9atL1xfsVOnTmrYsKF69+6tSZMmKScnRyNHjlRSUhKjlS9hdJY6U+j5Z7uEUuTrW0nVqkepWvUoNY5pqoe7ddHK5UvV+4l+yvzic235z0Z9+OlW2wjx+g1G6cvPt+rDlcvVu08/k3sPAOUQvzC5PaOz1Lkil/i5DddQVFioF55LVk72Ub32+hxbZpKk0P+7vm+t2nVs27y9vVW1WjUdy8ku874C5VmlSpVUIypKNaKi1KRpMz3YpZOWL12svv2eNrtrKGtkKYe4XHrIy8tTQUGB3baAgIBrtL76dXPyikqla3CC1WpV2rix+mT9Ws3KeEvVqlW/ZtuL06A/37ZVp06dVLv2Hcqqm4Bbi27QSF5eXvpy+zbbyMCDB37UsZxsNW7SzNzO4aY3depUeXh4qGfPnsrPz1dcXJzdUkOenp5auXKlEhMTFRsbKz8/P8XHx2vMmDEm9tq1GZGl8s8SpsqTkhKrCgov/J3n5f0mSbJctkKBxcNDVqu1zPsGAEB5Y0SWKjpXXCp9w593sVB3+NBBvfZGhgIvWY5fuvD92dvbWwcPHlDT5i1tj8k+elQRkVVN6DHgPkqsJVd8vgL4nUsU686dO6cRI0Zo4cKFOnny5BX7i4sJOeXV+LEv6cPVKzXt1enyq+Snn//vOnSV/f1VsWJFSdLyZUtUu3Yd3XJLsL7++r+alDZevR5PUM1atc3sOuAyzp8/p8M/HbLdzz5yWN9l7VFAQKAiIqvqzOlc5eRk286vQwcOSJJCQkIVEhqmyv7+erBbT70yeZICAgLl51dZUyaNV+MmzdS4SVMzXhLKmMWFhjBt2LDB7n7FihWVnp6u9PT0az4mKipKq1evLuWelW9kKfdx/vw5Hbn0M//oYX2ftUf+AYEKDArSvFlv6M627RUaGqbc3F+0dOG7+vnEMbXvGCdJahzTTP7+ARo3+nkl9Eu8sAzmssXKPnJYsXe1MetlAS7v/LlzOnTo93PvyOHD2rtnjwIDAxVZlR9nb3aulKVQOshS7uPy789HjxyxfX8ODQ3T88MHKWvvHv3Py9NVUlyskz9f+B4dEBioChW85Ve5srr1fEhvznxN4eERioisqvnzZkuSOtwbZ8prAlzRH2WnwKAgvfnGTLVr30GhYWHK/eUXLXh3vo4fO6Z74zqb2GuYhSzlGIvVBYbYJiUl6dNPP9XYsWPVu3dvpaen68iRI3r99dc1YcIEPfbYYzd0PGbWuY6mjepfdfuYf6Wpa/cekqRpU/5HHyxfptOnT6vqrbfq//3jYfWOT7C7uDPMdy6fE8ssX325XQOe6nPF9vse7KqRL43Xqg+WaVzqyCv2P/FUfz35TJKkCxfOfnXKJK1ds1qFBYVqFXunhqaMVEgo1y4yU4hf2YyZ2f7DaUOP95fagYYeD3+e0VnqBDPrTPPVl9v1z6ev/Mzv8kBXDX1+tF56Ybi+3fWNTuf+ooDAIDVo1FjxfZ9Wg0YxtrZ7v92lN9Jf1t49u1VUVKRatesqoV+iYu+8uyxfCi7jX9ElxkniGr7Y/rme7PP4Fdv/1rW7xo6fYEKP4IiyOq3MylKpqalXXAutfv362rt3r6QLM8CGDBmiBQsW2K1QEB4ebmt/6NAhJSYm6tNPP1XlypUVHx+vtLQ0eXnxmXQpo7PUKWbWmearL7cr6amEK7bf92A3Pfl0kno8cO9VH5f+RoZa3P4XSRdm0k1/bao+WrVC+fl5atS4iQYNfU6169xWml3HdVTyYal+V/JH2Wnk6Jf03PAh2vnN18r95RcFBQWpUeMY9Xs6UY1jmpjQW1yLu2ep8sYlinU1atTQvHnz1K5dOwUEBOirr75S3bp19dZbb+ndd9+94dH0FOsA41GsA4xHsQ5GMTpLUawDjEexDjCeu//AlJqaqsWLF2vdunW2bV5eXgoNvXDd6cTERK1atUoZGRkKDAzUgAED5OHhoc2bN0u6MBusWbNmioiI0L///W9lZ2fr8ccfV79+/TR+/HhDX1N5Z3SWolgHGI9iHWC8myFLlaeBTx6GH9EJp06dUu3aF5Y8DAgI0KlTpyRJd911lzZt2mRm1wAAKPcsBt/geshSAACUHjOzlJeXlyIiImy3i4W606dPa9asWZoyZYo6dOigli1bas6cOdqyZYu2bdsmSfr444/17bff6u2331azZs3UpUsXjR07Vunp6Vwz6DJkKQAASo+ZWapRo0bKzs623T777DPbvsGDB2vFihVatGiRNm7cqKNHj6pHjx62/cXFxbr//vtVUFCgLVu2aO7cucrIyNCoUaOceRuuyyWKdbVr19aPP/4oSYqOjtbChQslSStWrFDQZRd6BQAAN4hqndsjSwEAUIoMzlL5+fk6c+aM3S0/P/+qT/3999+ratWqql27th577DHb9YEyMzNVWFiojh072tpGR0erRo0a2rp1qyRp69atiomJsRsdHhcXpzNnzmj37t1GvTtugSwFAEApMvF3qfI08MklinV9+vTR119/LUl67rnnlJ6erooVK2rw4MEaNmyYyb0DAABwbWQpAADKj7S0NAUGBtrd0tLSrmjXqlUrZWRk6KOPPtKMGTP0448/6u6779avv/6qnJwceXt7X1FICg8PV05OjiQpJyfHrlB3cf/FffgdWQoAgPLDXQc+ucSFEwYPHmz7c8eOHbV3715lZmaqbt26atKEi04CAPBnWJgO5/bIUgAAlB6js1RKSoqSk5Pttvn4+FzRrkuXLrY/N2nSRK1atVJUVJQWLlwoX19fQ/t0syNLAQBQeozOUmlpaVdci2706NFKTU2123Zx4FP9+vWVnZ2tl156SXfffbd27drlkgOfTC/WFRYWqnPnzpo5c6Zuu+02SVJUVJSioqJM7hkAAO7BQq3OrZGlAAAoXUZnKR8fn6sW564nKChI9erV0759+3TvvfeqoKBAubm5dj8yHTt2TBEREZKkiIgIbd++3e4Yx44ds+3DBWQpAABKl9FZyl0HPpm+DGaFChX0zTffmN0NAACAcoksBQDAzeHs2bPav3+/IiMj1bJlS1WoUEHr16+37c/KytKhQ4cUGxsrSYqNjdXOnTt1/PhxW5u1a9cqICBADRs2LPP+uyqyFAAA5YuPj48CAgLsbo4MhLp04FNERIRt4NOlLh/4dHGg06X7L+4zmunFOknq1auXZs2aZXY3AABwSyZexxdlhCwFAEDpMStLDR06VBs3btSBAwe0ZcsWde/eXZ6ennrkkUcUGBiovn37Kjk5WZ9++qkyMzPVp08fxcbGqnXr1pKkTp06qWHDhurdu7e+/vprrVmzRiNHjlRSUpJTM/vcGVkKAIDS4yq/S7n6wCfTl8GUpKKiIs2ePVvr1q1Ty5Yt5efnZ7d/ypQpJvUMAAA3QIXN7ZGlAAAoRSZlqcOHD+uRRx7RyZMnFRYWprvuukvbtm1TWFiYJGnq1Kny8PBQz549lZ+fr7i4OE2fPt32eE9PT61cuVKJiYmKjY2Vn5+f4uPjNWbMGHNekAsjSwEAUIpMylJDhw7Vgw8+qKioKB09elSjR4++6sCn4OBgBQQEaODAgdcc+DRp0iTl5OSU6sAni9VqtRp+VAf98MMPqlmzpu65555rtrFYLPrkk09u6Lh5RX+2ZwAudy6fEwswWohf2YyZ+ergGUOP1yIqwNDjwXmllaVOnOUzHzCaf0WXGCcJuJWyOq3IUu6rtLLUqXPFf7ZrAC5TycfT7C4Absfds9TDDz+sTZs22Q18GjdunOrUqSNJysvL05AhQ/Tuu+/aDXy6dInLgwcPKjExURs2bLANfJowYYK8vIx/80wt1nl6eio7O1tVqlSRJD300EN65ZVXFB4e/qeOS7EOMB7FOsB4ZVWs++/BXw09XvMof0OPB+eVVpaiWAcYj2IdYLyyOq3IUu6rtLIUxTrAeBTrAOORpVyLqd8YL68Tfvjhhzp37pxJvQEAwD1ZWAbTbZGlAAAofWQp90WWAgCg9JGlHONhdgcuZeIkPwAAgHKPLAUAAOA8shQAADCLqTPrLBaLLJeVVS+/DwAA/hz+ZXVfZCkAAEof/7K6L7IUAAClj39ZHWP6MpgJCQny8fGRdOGCfs8884z8/Pzs2i1dutSM7gEA4B5IRW6LLAUAQBkgS7ktshQAAGWALOUQU4t18fHxdvd79eplUk8AAADKH7IUAACA88hSAADAVVisbrggd16R2T0A3M+5fE4swGghfmUzZuabn84aerwm1Ssbejy4nhNn+cwHjOZf0dRxkoBbKqvTiiyFG3XqXLHZXQDcTiUfT7O7ALgdspRr4RsjAABujstuAAAAOI8sBQAA4DyylGM8zO4AAAAAAAAAAAAAcLNiZh0AAG6OAUwAAADOI0sBAAA4jyzlGIp1AAC4O1IRAACA88hSAAAAziNLOYRlMAEAAAAAAAAAAACTMLMOAAA3Z2EIEwAAgNPIUgAAAM4jSzmGYh0AAG7OQiYCAABwGlkKAADAeWQpx7AMJgAAAAAAAAAAAGASZtYBAODmGMAEAADgPLIUAACA88hSjqFYBwCAuyMVAQAAOI8sBQAA4DyylENYBhMAAAAAAAAAAAAwCTPrAABwcxaGMAEAADiNLAUAAOA8spRjKNYBAODmLGQiAAAAp5GlAAAAnEeWcgzLYAIAAAAAAAAAAAAmYWYdAABujgFMAAAAziNLAQAAOI8s5RiKdQAAuDtSEQAAgPPIUgAAAM4jSzmEZTABAAAAAAAAAAAAkzCzDgAAN2dhCBMAAIDTyFIAAADOI0s5hmIdAABuzkImAgAAcBpZCgAAwHlkKcewDCYAAAAAAAAAAABgEmbWAQDg5hjABAAA4DyyFAAAgPPIUo6hWAcAgLsjFQEAADiPLAUAAOA8spRDWAYTAAAAAAAAAAAAMAkz6wAAcHMWhjABAAA4jSwFAADgPLKUY5hZBwCAm7NYjL05Ki0tTXfccYf8/f1VpUoVdevWTVlZWXZt8vLylJSUpJCQEFWuXFk9e/bUsWPH7NocOnRI999/vypVqqQqVapo2LBhKioqMuKtAQAAuC6zshQAAIA7IEs5hmIdAAAoFRs3blRSUpK2bdumtWvXqrCwUJ06ddK5c+dsbQYPHqwVK1Zo0aJF2rhxo44ePaoePXrY9hcXF+v+++9XQUGBtmzZorlz5yojI0OjRo0y4yUBAAAAAAAAhqNYBwCAm7MYfHPURx99pISEBDVq1EhNmzZVRkaGDh06pMzMTEnS6dOnNWvWLE2ZMkUdOnRQy5YtNWfOHG3ZskXbtm2TJH388cf69ttv9fbbb6tZs2bq0qWLxo4dq/T0dBUUFPzZtwYAAOC6zMpSjqxS0K5dO1ksFrvbM888Y9eGVQoAAICZzMpS5Q3FOgAA3J3BqSg/P19nzpyxu+Xn51+3G6dPn5YkBQcHS5IyMzNVWFiojh072tpER0erRo0a2rp1qyRp69atiomJUXh4uK1NXFyczpw5o927dzv9lgAAADjMpF+YHFmlQJL69eun7Oxs223SpEm2faxSAAAATEe1ziEU6wAAwA1JS0tTYGCg3S0tLe0PH1NSUqJBgwbpzjvvVOPGjSVJOTk58vb2VlBQkF3b8PBw5eTk2NpcWqi7uP/iPgAAAHd1vVUKLqpUqZIiIiJst4CAANs+VikAAAA3q/K2SgHFOgAA3JzF4P9SUlJ0+vRpu1tKSsof9iEpKUm7du3SggULyuhVAwAAGMPoLGXUKgUXzZ8/X6GhoWrcuLFSUlJ0/vx52z5WKQAAAGYzOks5qrytUuBl+BEBAIBLsRi8RICPj498fHwcbj9gwACtXLlSmzZtUrVq1WzbIyIiVFBQoNzcXLvZdceOHVNERIStzfbt2+2Od+zYMds+AACA0mZ0lkpLS9NLL71kt2306NFKTU295mOutkqBJD366KOKiopS1apV9c0332jEiBHKysrS0qVLJbFKAQAAMJ/RWSo/P/+KgU5X+63qo48+srufkZGhKlWqKDMzU23atLFtv7hKwdVcXKVg3bp1Cg8PV7NmzTR27FiNGDFCqamp8vb2NuhVMbMOAACUEqvVqgEDBmjZsmX65JNPVKtWLbv9LVu2VIUKFbR+/XrbtqysLB06dEixsbGSpNjYWO3cuVPHjx+3tVm7dq0CAgLUsGHDsnkhAAAABjJylYKnnnpKcXFxiomJ0WOPPaZ58+Zp2bJl2r9/f2m+BAAAANM4c3kWyfVXKWBmHQAAbs6sa+8mJSXpnXfe0fvvvy9/f3/b6O3AwED5+voqMDBQffv2VXJysoKDgxUQEKCBAwcqNjZWrVu3liR16tRJDRs2VO/evTVp0iTl5ORo5MiRSkpKuqHZfQAAAM4yOksZtUrB1bRq1UqStG/fPtWpU4dVCgAAgOmMzlIpKSlKTk6223a9bFUeVimgWAcAgJszerkBR82YMUPShYv1XmrOnDlKSEiQJE2dOlUeHh7q2bOn8vPzFRcXp+nTp9vaenp6auXKlUpMTFRsbKz8/PwUHx+vMWPGlNXLAAAANzmzspTVatXAgQO1bNkybdiw4YpVCq5mx44dkqTIyEhJF1YpGDdunI4fP64qVapIYpUCAABQtsy+PIv0+yoFn332md32p556yvbnmJgYRUZG6p577tH+/ftVp04dQ/rrKIp1AACgVFit1uu2qVixotLT05Wenn7NNlFRUVq9erWRXQMAAHB511ulYP/+/XrnnXd03333KSQkRN98840GDx6sNm3aqEmTJpJYpQAAAKC8rFLANesAAHB7FoNvAAAANxNzstSMGTN0+vRptWvXTpGRkbbbe++9J0ny9vbWunXr1KlTJ0VHR2vIkCHq2bOnVqxYYTvGxVUKPD09FRsbq169eunxxx9nlQIAAFCGzMlSVqtVAwYM0LJly/TJJ584vUrBzp07dfz4cVub0lqlwGJ1ZNh7OZNXZHYPAPdzLp8TCzBaiF/ZTHA/kltg6PFuDfI29HhwPSfO8pkPGM2/IouaAEYrq9OKLIUbdepcsdldANxOJR9Ps7sAuB13z1L9+/e3rVJQv3592/brrVJQrVo1bdy4UZJUXFysZs2aqWrVqrZVCnr37q0nn3xS48ePN/R1UawD4BCKdYDxKNbBVVGsA4xHsQ4wnrv/wITyi2IdYDyKdYDx3D1LWa5xsbw5c+YoISFBP/30k3r16qVdu3bp3Llzql69urp3766RI0cqICDA1v7gwYNKTEzUhg0b5Ofnp/j4eE2YMEFeXsa+gRTrADiEYh1gvLIq1h01OBRV5Qcmt0exDjAexTrAeGV1WpGlcKMo1gHGo1gHGI8s5Vr4xggAgJu7xkAiAAAAOIAsBQAA4DyylGM8zO4AAAAAAAAAAAAAcLNiZh0AAG7OIoYwAQAAOIssBQAA4DyylGMo1gEA4O7IRAAAAM4jSwEAADiPLOUQlsEEAAAAAAAAAAAATMLMOgAA3BwDmAAAAJxHlgIAAHAeWcoxFOsAAHBzFlIRAACA08hSAAAAziNLOYZlMAEAAAAAAAAAAACTMLMOAAA3Z2HBAQAAAKeRpQAAAJxHlnIMxToAANwdmQgAAMB5ZCkAAADnkaUcwjKYAAAAAAAAAAAAgEmYWQcAgJtjABMAAIDzyFIAAADOI0s5hmIdAABuzkIqAgAAcBpZCgAAwHlkKcewDCYAAAAAAAAAAABgEmbWAQDg5iwsOAAAAOA0shQAAIDzyFKOoVgHAICbY7kBAAAA55GlAAAAnEeWcgzLYAIAAAAAAAAAAAAmoVgHAAAAAAAAAAAAmIRlMAEAcHMsNwAAAOA8shQAAIDzyFKOYWYdAAAAAAAAAAAAYBJm1gEA4OYsYggTAACAs8hSAAAAziNLOYZiHQAAbo7lBgAAAJxHlgIAAHAeWcoxLIMJAAAAAAAAAAAAmISZdQAAuDkGMAEAADiPLAUAAOA8spRjKNYBAODuSEUAAADOI0sBAAA4jyzlEJbBBAAAAAAAAAAAAEzCzDoAANychSFMAAAATiNLAQAAOI8s5RiKdQAAuDkLmQgAAMBpZCkAAADnkaUcwzKYAAAAAAAAAAAAgEmYWQcAgJtjABMAAIDzyFIAAADOI0s5hmIdAADujlQEAADgPLIUAACA88hSDmEZTAAAAAAAAAAAAMAkzKwDAMDNWRjCBAAA4DSyFAAAgPPIUo6hWAcAgJuzkIkAAACcRpYCAABwHlnKMSyDCQAAAAAAAAAAAJjEYrVarWZ3Ajen/Px8paWlKSUlRT4+PmZ3B3ALnFcAcPPgMx8wHucVANw8+MwHjMd5BTiPYh1Mc+bMGQUGBur06dMKCAgwuzuAW+C8AoCbB5/5gPE4rwDg5sFnPmA8zivAeSyDCQAAAAAAAAAAAJiEYh0AAAAAAAAAAABgEop1AAAAAAAAAAAAgEko1sE0Pj4+Gj16NBcbBQzEeQUANw8+8wHjcV4BwM2Dz3zAeJxXgPMsVqvVanYnAAAAAAAAAAAAgJsRM+sAAAAAAAAAAAAAk1CsAwAAAAAAAAAAAExCsQ4AAAAAAAAAAAAwCcU63LRq1qypadOmmd0NwOUkJCSoW7duZncDAODiyFLA1ZGlAACOIEsBV0eWws2KYt1NLCEhQRaLRRMmTLDbvnz5clkslhs6lqMBo2bNmrJYLHa3atWq3dBzAe7q4jl5+W3fvn1mdw0AcBVkKcC1kKUAoHwhSwGuhSwFmIti3U2uYsWKmjhxon755Zcye84xY8YoOzvbdvvvf/971XaFhYVl1ifAVXTu3Nnu/MjOzlatWrXs2hQUFJjUOwDA5chSgGshSwFA+UKWAlwLWQowD8W6m1zHjh0VERGhtLS0P2y3ZMkSNWrUSD4+PqpZs6YmT55s29euXTsdPHhQgwcPto24+CP+/v6KiIiw3cLCwiRJFotFM2bM0N/+9jf5+flp3LhxKi4uVt++fVWrVi35+vqqfv36evnll+2O165dOw0aNMhuW7du3ZSQkGC7f/z4cT344IPy9fVVrVq1NH/+fAfeHaDs+fj42J0fERERuueeezRgwAANGjRIoaGhiouLkyRNmTJFMTEx8vPzU/Xq1dW/f3+dPXvWdqzU1FQ1a9bM7vjTpk1TzZo1bfeLi4uVnJysoKAghYSEaPjw4bJarWXxUgHALZClANdClgKA8oUsBbgWshRgHop1NzlPT0+NHz9er776qg4fPnzVNpmZmfrHP/6hhx9+WDt37lRqaqpefPFFZWRkSJKWLl2qatWq2Y1MclZqaqq6d++unTt36oknnlBJSYmqVaumRYsW6dtvv9WoUaP0/PPPa+HChTd03ISEBP3000/69NNPtXjxYk2fPl3Hjx93up9AWZs7d668vb21efNmzZw5U5Lk4eGhV155Rbt379bcuXP1ySefaPjw4Td03MmTJysjI0OzZ8/WZ599plOnTmnZsmWl8RIAwC2RpYDygSwFAK6JLAWUD2QpoAxYcdOKj4+3du3a1Wq1Wq2tW7e2PvHEE1ar1WpdtmyZ9dL/NR599FHrvffea/fYYcOGWRs2bGi7HxUVZZ06dep1nzMqKsrq7e1t9fPzs91efvllq9VqtUqyDho06LrHSEpKsvbs2dN2v23bttZnn33Wrk3Xrl2t8fHxVqvVas3KyrJKsm7fvt22f8+ePVZJDvUZKCvx8fFWT09Pu/Pj73//u7Vt27bW5s2bX/fxixYtsoaEhNjujx492tq0aVO7NlOnTrVGRUXZ7kdGRlonTZpku19YWGitVq2a7bMBAHBtZCmyFFwLWQoAyheyFFkKroUsBZjLy5wSIVzNxIkT1aFDBw0dOvSKfXv27FHXrl3ttt15552aNm2aiouL5enpeUPPNWzYMLulAEJDQ21/vv32269on56ertmzZ+vQoUP67bffVFBQcMUU6j+yZ88eeXl5qWXLlrZt0dHRCgoKuqF+A2Whffv2mjFjhu2+n5+fHnnkEbv/fy9at26d0tLStHfvXp05c0ZFRUXKy8vT+fPnValSpes+1+nTp5Wdna1WrVrZtnl5een2229nyQEAuEFkKcA1kKUAoHwiSwGugSwFmIdlMCFJatOmjeLi4pSSklLqzxUaGqq6devabpeGEz8/P7u2CxYs0NChQ9W3b199/PHH2rFjh/r06WN3IVMPD48rPsC5CDDKKz8/P7vzIzIy0rb9UgcOHNADDzygJk2aaMmSJcrMzFR6erqk3y/0y7kBAGWHLAW4BrIUAJRPZCnANZClAPNQrIPNhAkTtGLFCm3dutVue4MGDbR582a7bZs3b1a9evVso5e8vb1VXFxseJ82b96sv/71r+rfv7+aN2+uunXrav/+/XZtwsLC7NYjLy4u1q5du2z3o6OjVVRUpMzMTNu2rKws5ebmGt5foKxkZmaqpKREkydPVuvWrVWvXj0dPXrUrk1YWJhycnLsgtGOHTtsfw4MDFRkZKQ+//xz27bLzxUAgOPIUkD5QZYCANdDlgLKD7IUYDyKdbCJiYnRY489pldeecVu+5AhQ7R+/XqNHTtW3333nebOnavXXnvNbmmCmjVratOmTTpy5Ih+/vlnw/p022236csvv9SaNWv03Xff6cUXX9QXX3xh16ZDhw5atWqVVq1apb179yoxMdEu8NSvX1+dO3fW008/rc8//1yZmZl68skn5evra1g/gbJWt25dFRYW6tVXX9UPP/ygt956y3aB34vatWunEydOaNKkSdq/f7/S09P14Ycf2rV59tlnNWHCBC1fvlx79+5V//79+cIAAE4iSwHlB1kKAFwPWQooP8hSgPEo1sHOmDFjVFJSYretRYsWWrhwoRYsWKDGjRtr1KhRGjNmjN363mPGjNGBAwdUp04dhYWFGdafp59+Wj169NBDDz2kVq1a6eTJk+rfv79dmyeeeELx8fF6/PHH1bZtW9WuXVvt27e3azNnzhxVrVpVbdu2VY8ePfTUU0+pSpUqhvUTKGtNmzbVlClTNHHiRDVu3Fjz589XWlqaXZsGDRpo+vTpSk9PV9OmTbV9+/Yr1v8fMmSIevfurfj4eMXGxsrf31/du3cvy5cCAG6FLAWUD2QpAHBNZCmgfCBLAcazWLlaIwAAAAAAAAAAAGAKZtYBAAAAAAAAAAAAJqFYBwAAAAAAAAAAAJiEYh0AAAAAAAAAAABgEop1AAAAAAAAAAAAgEko1gEAAAAAAAAAAAAmoVgHAAAAAAAAAAAAmIRiHQAAAAAAAAAAAGASinUAAAAAAAAAAACASSjWAbhCQkKCunXrZrvfrl07DRo0qMz7sWHDBlksFuXm5pb5cwMAADiLLAUAAOA8shSAmxHFOqAcSUhIkMVikcVikbe3t+rWrasxY8aoqKioVJ936dKlGjt2rENtCTIAAMBVkaUAAACcR5YCgNLjZXYHANyYzp07a86cOcrPz9fq1auVlJSkChUqKCUlxa5dQUGBvL29DXnO4OBgQ44DAABgNrIUAACA88hSAFA6mFkHlDM+Pj6KiIhQVFSUEhMT1bFjR33wwQe2JQLGjRunqlWrqn79+pKkn376Sf/4xz8UFBSk4OBgde3aVQcOHLAdr7i4WMnJyQoKClJISIiGDx8uq9Vq95yXLzeQn5+vESNGqHr16vLx8VHdunU1a9YsHThwQO3bt5ck3XLLLbJYLEpISJAklZSUKC0tTbVq1ZKvr6+aNm2qxYsX2z3P6tWrVa9ePfn6+qp9+/Z2/QQAADACWQoAAMB5ZCkAKB0U64ByztfXVwUFBZKk9evXKysrS2vXrtXKlStVWFiouLg4+fv76z//+Y82b96sypUrq3PnzrbHTJ48WRkZGZo9e7Y+++wznTp1SsuWLfvD53z88cf17rvv6pVXXtGePXv0+uuvq3LlyqpevbqWLFkiScrKylJ2drZefvllSVJaWprmzZunmTNnavfu3Ro8eLB69eqljRs3SroQ3nr06KEHH3xQO3bs0JNPPqnnnnuutN42AAAASWQpAACAP4MsBQDGYBlMoJyyWq1av3691qxZo4EDB+rEiRPy8/PTm2++aVtm4O2331ZJSYnefPNNWSwWSdKcOXMUFBSkDRs2qFOnTpo2bZpSUlLUo0cPSdLMmTO1Zs2aaz7vd999p4ULF2rt2rXq2LGjJKl27dq2/ReXJqhSpYqCgoIkXRjxNH78eK1bt06xsbG2x3z22Wd6/fXX1bZtW82YMUN16tTR5MmTJUn169fXzp07NXHiRAPfNQAAgAvIUgAAAM4jSwGAsSjWAeXMypUrVblyZRUWFqqkpESPPvqoUlNTlZSUpJiYGLv1wL/++mvt27dP/v7+dsfIy8vT/v37dfr0aWVnZ6tVq1a2fV5eXrr99tuvWHLgoh07dsjT01Nt27Z1uM/79u3T+fPnde+999ptLygoUPPmzSVJe/bsseuHJFuAAgAAMApZCgAAwHlkKQAoHRTrgHKmffv2mjFjhry9vVW1alV5ef1+Gvv5+dm1PXv2rFq2bKn58+dfcZywsDCnnt/X1/eGH3P27FlJ0qpVq3Trrbfa7fPx8XGqHwAAAM4gSwEAADiPLAUApYNiHVDO+Pn5qW7dug61bdGihd577z1VqVJFAQEBV20TGRmpzz//XG3atJEkFRUVKTMzUy1atLhq+5iYGJWUlGjjxo225QYudXEEVXFxsW1bw4YN5ePjo0OHDl1z5FODBg30wQcf2G3btm3b9V8kAADADSBLAQAAOI8sBQClw8PsDgAoPY899phCQ0PVtWtX/ec//9GPP/6oDRs26J///KcOHz4sSXr22Wc1YcIELV++XHv37lX//v2Vm5t7zWPWrFlT8fHxeuKJJ7R8+XLbMRcuXChJioqKksVi0cqVK3XixAmdPXtW/v7+Gjp0qAYPHqy5c+dq//79+uqrr/Tqq69q7ty5kqRnnnlG33//vYYNG6asrCy98847ysjIKO23CAAA4JrIUgAAAM4jSwGA4yjWAW6sUqVK2rRpk2rUqKEePXqoQYMG6tu3r/Ly8mwjmoYMGaLevXsrPj5esbGx8vf3V/fu3f/wuDNmzNDf//539e/fX9HR0erXr5/OnTsnSbr11lv10ksv6bnnnlN4eLgGDBggSRo7dqxefPFFpaWlqUGDBurcubNWrVqlWrVqSZJq1KihJUuWaPny5WratKlmzpyp8ePHl+K7AwAA8MfIUgAAAM4jSwGA4yzWa12tEwAAAAAAAAAAAECpYmYdAAAAAAAAAAAAYBKKdQAAAAAAAAAAAIBJKNYBAAAAAAAAAAAAJqFYBwAAAAAAAAAAAJiEYh0AAAAAAAAAAABgEop1AAAAAAAAAAAAgEko1gEAAAAAAAAAAAAmoVgHAAAAAAAAAAAAmIRiHQAAAAAAAAAAAGASinUAAAAAAAAAAACASSjWAQAAAAAAAAAAACb5/8L68tbs8R6hAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAxYAAAJOCAYAAAAqFJGJAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA24lJREFUeJzs3Xd4FOXXxvHvphdqhNCbgBTpVbCBoICKIApIDagIIlgoAlIERZEiooJEQQQCiICovCKo4C8qVaqCSJHei5RQQtrO+8fCkkASssluZndzf64rV2ZnZ2dOSEjm7HOe81gMwzAQERERERHJAh+zAxAREREREc+nxEJERERERLJMiYWIiIiIiGSZEgsREREREckyJRYiIiIiIpJlSixERERERCTLlFiIiIiIiEiWKbEQEREREZEsU2IhIiIiIiJZpsRCREQkmUaNGtGoUSP74wMHDmCxWJg5c6ZpMd1O6dKl6datW6Zea7FYGDlypFPjEZGcSYmFiOQYM2fOxGKx2D/8/PwoVqwY3bp14+jRo6m+xjAMoqKieOCBB8iXLx8hISFUrVqVt956i8uXL6d5rW+++YYWLVpQoEABAgICKFq0KO3ateOXX37JUKxXr17lgw8+oH79+uTNm5egoCDuuusu+vTpw+7duzP19Qv8888/WCwWgoKCOH/+vFPPHR0dbf/ZmjNnTqrH3HvvvVgsFqpUqeLUa4uIuAMlFiKS47z11ltERUURGRlJixYtmDNnDg8++CBXr15NcVxSUhLPPPMMXbt2BWDkyJFMmjSJGjVqMGrUKO655x5OnjyZ4jWGYdC9e3fatGnDyZMn6devH5GRkbz00kvs27ePJk2asGbNmnTjO3PmDPfddx/9+vUjPDyct956iylTptC6dWuWLFmim9IsmDNnDoULFwZg0aJFLrlGUFAQ8+bNu2X/gQMHWLNmDUFBQS65roiI2fzMDkBEJLu1aNGCOnXqAPD8889ToEABxo4dy5IlS2jXrp39uHHjxrFgwQIGDBjA+PHj7ftfeOEF2rVrR+vWrenWrRvLli2zP/f+++8zc+ZMXn31VSZOnIjFYrE/N3ToUKKiovDzS/9Xb7du3diyZQuLFi3iqaeeSvHc22+/zdChQ7P09V+XmJiI1WolICDAKedzd4ZhMG/ePDp27Mj+/fuZO3cuzz//vNOv8+ijj7JkyRLOnDlDgQIF7PvnzZtHoUKFKF++POfOnXP6dUVEzKYRCxHJ8e6//34A9u7da98XGxvL+PHjueuuuxgzZswtr2nZsiUREREsX76cdevW2V8zZswYKlasyIQJE1IkFdd16dKFevXqpRnL+vXrWbp0Kc8999wtSQVAYGAgEyZMsD++eT7Add26daN06dL2x9fnCUyYMIFJkyZRtmxZAgMD2bJlC35+fowaNeqWc+zatQuLxcLkyZPt+86fP8+rr75KiRIlCAwMpFy5cowdOxar1ZritfPnz6d27drkzp2bPHnyULVqVT788MM0v+7ssHr1ag4cOMAzzzzDM888w2+//caRI0ecfp1WrVoRGBjIwoULU+yfN28e7dq1w9fX95bXJCYm8vbbb9u/L6VLl+aNN94gLi4uxXGGYTB69GiKFy9OSEgIjRs35u+//041jox+r2528eJFXn31VUqXLk1gYCDh4eE8/PDDbN682cF/CRHJaZRYiEiOd+DAAQDy589v37dq1SrOnTtHx44d0xxhuF4i9f3339tfc/bsWTp27JjqzWNGLFmyBLAlIK7wxRdf8PHHH/PCCy/w/vvvU6RIER588EEWLFhwy7FfffUVvr6+tG3bFoArV67w4IMPMmfOHLp27cpHH33Evffey5AhQ+jXr5/9dT///DMdOnQgf/78jB07lvfee49GjRqxevVql3xNGTV37lzKli1L3bp1admyJSEhIXz55ZdOv05ISAitWrVKce4///yTv//+m44dO6b6mueff54RI0ZQq1YtPvjgAx588EHGjBnDM888k+K4ESNGMHz4cKpXr8748eO58847eeSRR26Z75PR71VqevXqxdSpU3nqqaf45JNPGDBgAMHBwfzzzz+Z/BcRkZxCpVAikuNcuHCBM2fOcPXqVdavX8+oUaMIDAzk8ccftx+zY8cOAKpXr57mea4/d/2G6/rnqlWrZjo2Z5wjPUeOHOHff/+lYMGC9n3t27enZ8+ebN++PcX8ja+++ooHH3yQQoUKATBx4kT27t3Lli1bKF++PAA9e/akaNGijB8/nv79+1OiRAmWLl1Knjx5+PHHHzOdYDlbQkICCxcupFevXgAEBwfzxBNPMHfuXAYOHOj063Xs2JGWLVty+PBhSpQowdy5c7nzzju55557bjn2zz//ZNasWTz//PNMmzYNgN69exMeHs6E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PvuM8+fP8+CDD/LHH38wa9YsWrduTePGjQHbXIoBAwYwZswYHn/8cR599FG2bNnCsmXLbhmJyej36mYXL16kePHiPP3001SvXp1cuXKxYsUKNmzYkKJLlIhIqszObEREskt6IxZJSUlG2bJljbJlyxqJiYkp9n/xxRfGvffea+TJk8cICgoy7r77bmPUqFHGpUuX0rzWokWLjEceecQICwsz/Pz8jCJFihjt27c3oqOjMxTrlStXjAkTJhh169Y1cuXKZQQEBBjly5c3+vbta/z7778pjp0zZ45x5513GgEBAUaNGjWMH3/80YiIiHB4xCImJsYIDg42AGPOnDmpHnPx4kVjyJAhRrly5YyAgACjQIECRsOGDY0JEyYY8fHxKb728PBwIyAgwChZsqTRs2dP4/jx4xn62p3p/fffNwBj5cqVaR4zc+ZMAzC+++47wzCcO2KRmptHLAzDMBISEoxRo0YZZcqUMfz9/Y0SJUoYQ4YMMa5evZriuKSkJGPUqFFGkSJFjODgYKNRo0bG9u3bjVKlSqUYsTCMjH2vDCPliEVcXJwxcOBAo3r16kbu3LmN0NBQo3r16sYnn3yS7tckImIYhmExjFRm/YmIiIiIiDhA61iIiIiIiEiWKbEQEREREZEsU2IhIiIiIiJZpsRCRERERESyTImFiIiIiIhkmRILERERERHJshy3QJ7VauXYsWPkzp0bi8VidjgiIiIiIm7LMAwuXrxI0aJF8fFJf0wixyUWx44do0SJEmaHISIiIiLiMQ4fPkzx4sXTPSbHJRa5c+cGbP84efLkMTkaERERERH3FRMTQ4kSJez30OnJcYnF9fKnPHnyKLEQEREREcmAjEwh0ORtERERERHJMiUWIiIiIiKSZUosREREREQky5RYiIiIiIhIlimxEBERERGRLFNiISIiIiIiWabEQkREREREskyJhYiIiIiIZJkSCxERERERyTIlFiIiIiIikmVKLEREREREJMuUWIiIiIiISJYpsRARERERkSxTYiEiIiIiIlmmxEJERERERLJMiYWIiIiIiGSZqYnFb7/9RsuWLSlatCgWi4Vvv/32tq+Jjo6mVq1aBAYGUq5cOWbOnOnyOEVEREREJH2mJhaXL1+mevXqTJkyJUPH79+/n8cee4zGjRuzdetWXn31VZ5//nl+/PFHF0cqIiIiIiLp8TPz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kWz333//0bNnTw4dOsTq1avx9/cnICCANm3amB2aiEi6HJ5jsX//fqckFfHx8WzatImmTZveCMbHh6ZNm7J27dpUX7NkyRIaNGjASy+9RKFChahSpQrvvvtuqnM+RBxmby17AYrXta1Z4cXUWlbE/fz0009UrVqVr7/+mi1btqT591BExB1lqubh8uXL/Prrrxw6dIj4+PgUz7388ssZOseZM2dISkqiUKFCKfYXKlSInTt3pvqaffv28csvv9CpUyd++OEH/v33X3r37k1CQgJvvvlmqq+Ji4sjLi7O/jgmJiZD8Yn3ur7advJVtgFY9QEcWmtrLdvGe1vLwq0lUGotK2Ku2NhYBg8ezEcffQRAxYoVmTNnDrVr1zY5MhGRjHP4zmnLli08+uijXLlyhcuXLxMWFsaZM2cICQkhPDw8w4lFZlitVsLDw/nss8/w9fWldu3aHD16lPHjx6eZWIwZM4ZRo0a5LCbxLGmutn10M0SPsW0/NgHCymR7bNkltRIotZYVMc/WrVvp1KkTO3bsAOCll15i3LhxhISE3OaVIiLuxeFSqNdee42WLVty7tw5goODWbduHQcPHqR27dpMmDAhw+cpUKAAvr6+nDx5MsX+kydPUrhw4VRfU6RIEe666y58fW+Ua1SqVIkTJ07cMnJy3ZAhQ7hw4YL94/DhwxmOUbxPaqtt1yxQjeDvettay1Z5Cqp5b2tZUAmUiDsxDIO+ffuyY8cOChcuzA8//MDkyZOVVIiIR3I4sdi6dSv9+/fHx8cHX19f4uLiKFGiBOPGjeONN97I8HkCAgKoXbs2K1eutO+zWq2sXLmSBg0apPqae++9l3///dc+kRxg9+7dFClShICAgFRfExgYSJ48eVJ8iIBtte31HdczKy43lnPXW8tOBC98595W+pR47SNlCdTCXg00WiFiEovFwueff06HDh3466+/aNGihdkhiYhkmsOlUP7+/vj42PKR8PBwDh06RKVKlcibN6/DowH9+vUjIiKCOnXqUK9ePSZNmsTly5ftXaK6du1KsWLFGDPGVqLy4osvMnnyZF555RX69u3Lnj17ePfdd11afiXewzAMIpbfaJUc7BdMyK5l8NeXttaybT6D4HzmBegi6XV/UgmUSPabN28e+/fvZ+jQoQDcddddzJs3z+SoRESyzuHEombNmmzYsIHy5cvz4IMP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MAxmz55N3759uXjxItu3b2fgwIH6myQiYjKHE4vVq1e7Ig6RzPHS+RUikrr//vuPnj178vXXXwNw7733EhUVpaRCRMQNZCixWLJkCS1atMDf358lS5ake+wTTzzhlMBEbsswboxYaH6FiNf76aef6NatG8ePH8fPz4+33nqL119/HV9ftWIWEXEHGUosWrduzYkTJwgPD6d169ZpHmexWEhKSkrzeRGnOn8QLh4HHz8oWsvsaLLMMIwUK22LyA3nz5+nbdu2xMTEULFiRebMmUPt2rXNDktERJLJUGJhtVpT3RYx1aFroxVFakBAiKmhZJVW3BZJX758+fjoo4/YsGED48aNIyTEs//Pi4h4I4dnu86ePZu4uLhb9sfHxzN79mynBCWSIV60MN7NK25rpW3J6ZKSkhg3bhwrVqyw74uIiGDy5MlKKkRE3JTDiUX37t25cOHCLfsvXrxI9+7dnRKUCFwrDUq4wpWEK8Qmxt56wKFkHaE8nGHc2N44rCkLezXQoniSYx06dIgmTZowaNAgIiIiiImJMTskERHJAIe7QhmGkeoNz5EjR8ibN69TghIxDIOuy7qy9fTW1A+IPQ+ndti2PXzEwjAM2kautT8OCfBVUiE51rx58+jduzcXLlwgNDSUt956i9y5c5sdloiIZECGE4uaNWtisViwWCw0adIEP78bL01KSmL//v1aIE+cJjYxNtWkomZ4Tduq2wfWAAbkLwO5wrM9PmeKTUhix3HbO7KVi+RRCZTkSOfOnaN3797Mnz8fgHvuuYeoqCjKlStncmQiIpJRGU4srneD2rp1K82aNSNXrlz25wICAihdujRPPfWU0wOUnMMwDHvJU/LSp+h20bZkAgj2C7a9m+/h8ysMwyA2wdYBKnknKJVASU506tQpateuzZEjR/D19WXEiBG88cYbKd7AEhER95fh39pvvvkmAKVLl6Z9+/YEBQW5LCjJedIrfQr2CybE/6bJmh68MF56HaCUU0hOVLBgQe677z42btzInDlzqF/f8/5fi4hIJuZYREREuCIOyeFuW/qUXFICHN1k2/bAEYubO0Bdp05QkpNs376dQoUKUbBgQSwWC5GRkfj6+qYYDRcREc+SocQiLCyM3bt3U6BAAfLnz59uqcbZs2edFpx4v+vlT7ctfUruxDZIuAJBeaFAhewMN8tuXgRv47CmhATY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y/Fllbo+iTc4c+YMPXr04NtvvwXgwQcfZNasWZQqVcrcwERExKM5NHl7+PDhhISE2B/Hx8fzzjvvkDdvXvu+iRMnOi868WhZLoECt5xfcZ26PomnunTpEitXrsTf35/Ro0fTv39/fH2VIIuISNZkOLF44IEH2LVrV4p9DRs2ZN++ffbHusHKua7Pq8jyCts3O3QtsTBxfkXytrLJ51ao65N4ksTERPtq2aVLl2bOnDmULFmSGjVqmBuYiIh4jQwnFtHR0S4MQzyZ01bXvtnGGXBglW27VEPnnjuDtJK2eINNmzbRpUsXJk2axCOPPALAE088YXJUIiLibTRDT7Ls5nkVkMW5FYYBv46H718DDKjbAwpWyHKcmZFWW1nNrRBPkJSUxJgxY7jnnnv4559/GDp0KIZhmB2WiIh4KYdX3hZJT3S7aIL9gjNfBmW1wvLB8MentscPvA6N33BukBmUXltZtZMVd7d//366du3KqlW2Ub+nn36ayMhI/dyKiIjLKLGQLDEMg4jlN9oQB/sFE+Ifks4r0pEYD9++CNsX2R63GAf1ezohSselVgJ1va2siDszDIOoqCj69OnDxYsXyZ07N5MnT6ZLly5KKkRExKV0lyRZEpsYy86zOwGoGFYx8+VPcZdgQRfY+wv4+MGTn0LVp50YqWPUVlY81apVq+xrDt17771ERUVRpkwZk6MSEZGcQImFOM2s5rMy947o5f9gXls4ugn8Q6B9FJRr6vwAbyOt7k9qKyue5P7776dbt26UL1+eQYMGqY2siIhkm0wlFr///juffvope/fuZdGiRRQrVsz+rth9993n7BjFm50/DHPawJndEJwfOi2C4nWyPYz0uj+pray4s9jYWEaPHs2rr75KwYIFAZgxY4Z+ZkVEJNs53BXq66+/plmzZgQHB7Nlyxbi4uIAuHDhAu+++67TAxQvdnoXzGhmSyryFINnfzQlqQB1fxLP9Oeff1K3bl3effddevToYd+vpEJERMzg8IjF6NGjiYyMpGvXrsyfP9++/95772X06NFODU682JGNMPdpiD0HBe6CzoshXwlTQlH3J/E0VquV999/n6FDh5KQkEChQoV44YUXzA5LRERyOIcTi127dvHAAw/csj9v3rycP3/eGTGJt9sXDV92gIQrUKw2dFwIoXeYEoq6P4mnOXToEBEREfZFS1u1asW0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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[evaluate_models] Evaluation complete.\n",
            "\n",
            "── Verification ──────────────────────────────────────\n",
            "\n",
            "logistic_regression:\n",
            "  accuracy    : 0.6775\n",
            "  precision   : 0.1330\n",
            "  recall      : 0.7914\n",
            "  f1          : 0.2277\n",
            "  auc_roc     : 0.7824\n",
            "  confusion_matrix:\n",
            "[[1457  717]\n",
            " [  29  110]]\n",
            "\n",
            "random_forest:\n",
            "  accuracy    : 0.9395\n",
            "  precision   : 0.3333\n",
            "  recall      : 0.0072\n",
            "  f1          : 0.0141\n",
            "  auc_roc     : 0.8378\n",
            "  confusion_matrix:\n",
            "[[2172    2]\n",
            " [ 138    1]]\n",
            "\n",
            "xgboost:\n",
            "  accuracy    : 0.9377\n",
            "  precision   : 0.4194\n",
            "  recall      : 0.0935\n",
            "  f1          : 0.1529\n",
            "  auc_roc     : 0.8565\n",
            "  confusion_matrix:\n",
            "[[2156   18]\n",
            " [ 126   13]]\n"
          ]
        }
      ],
      "source": [
        "result_state = evaluate_models(result_state)\n",
        "\n",
        "print()\n",
        "print('── Verification ──────────────────────────────────────')\n",
        "for model_name, metrics in result_state['eval_results'].items():\n",
        "    print(f'\\n{model_name}:')\n",
        "    for metric, value in metrics.items():\n",
        "        print(f'  {metric:<12}: {value:.4f}')\n",
        "    cm = result_state['conf_matrices'][model_name]\n",
        "    print(f'  confusion_matrix:\\n{cm}')"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "6d226364",
      "metadata": {
        "id": "6d226364"
      },
      "source": [
        "# Section 4 — Select-Model Agent Node (Utkarsh)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "id": "select_model_fn",
      "metadata": {
        "id": "select_model_fn"
      },
      "outputs": [],
      "source": [
        "def select_model(state: AgentState) -> AgentState:\n",
        "    \"\"\"\n",
        "    LangGraph node: selects the best model for fraud detection.\n",
        "\n",
        "    Criteria (tuned to Section 3 results):\n",
        "      1. AUC-ROC  >= 0.78\n",
        "      2. Recall   >= 0.70\n",
        "      3. F1-score >= 0.15\n",
        "    Among models that pass all three gates, the one with the\n",
        "    highest AUC-ROC is selected.\n",
        "    \"\"\"\n",
        "    eval_results   = state['eval_results']\n",
        "    trained_models = state['trained_models']\n",
        "\n",
        "    THRESH_AUC    = 0.78\n",
        "    THRESH_RECALL = 0.70\n",
        "    THRESH_F1     = 0.15\n",
        "\n",
        "    print('[select_model] ── Selection Criteria ──')\n",
        "    print(f'  AUC-ROC  >= {THRESH_AUC}')\n",
        "    print(f'  Recall   >= {THRESH_RECALL}')\n",
        "    print(f'  F1-score >= {THRESH_F1}')\n",
        "\n",
        "    # ── Step 1: screen every model ──────────────────────────────────────────\n",
        "    qualified = {}\n",
        "    for name, m in eval_results.items():\n",
        "        auc_ok    = m['auc_roc']  >= THRESH_AUC\n",
        "        recall_ok = m['recall']   >= THRESH_RECALL\n",
        "        f1_ok     = m['f1']       >= THRESH_F1\n",
        "        passed    = auc_ok and recall_ok and f1_ok\n",
        "\n",
        "        print(f'\\n[select_model] {name}:')\n",
        "        print(f'  AUC-ROC  = {m[\"auc_roc\"]:.4f}  {\"PASS\" if auc_ok    else \"FAIL\"}')\n",
        "        print(f'  Recall   = {m[\"recall\"]:.4f}  {\"PASS\" if recall_ok else \"FAIL\"}')\n",
        "        print(f'  F1-score = {m[\"f1\"]:.4f}  {\"PASS\" if f1_ok     else \"FAIL\"}')\n",
        "        print(f'  -> {\"QUALIFIED\" if passed else \"ELIMINATED\"}')\n",
        "\n",
        "        if passed:\n",
        "            qualified[name] = m\n",
        "\n",
        "    # ── Step 2: pick the winner ─────────────────────────────────────────────\n",
        "    if qualified:\n",
        "        best_name = max(qualified, key=lambda n: qualified[n]['auc_roc'])\n",
        "        best_metrics = qualified[best_name]\n",
        "    else:\n",
        "        # Fallback: no model meets all three — take highest AUC-ROC\n",
        "        print('\\n[select_model] No model met all three criteria.')\n",
        "        print('[select_model] Falling back to highest AUC-ROC among remaining.')\n",
        "        best_name = max(eval_results, key=lambda n: eval_results[n]['auc_roc'])\n",
        "        best_metrics = eval_results[best_name]\n",
        "\n",
        "    best_model = trained_models[best_name]\n",
        "\n",
        "    # ── Step 3: build justification string ──────────────────────────────────\n",
        "    justification = (\n",
        "        f\"Selected '{best_name}' — \"\n",
        "        f\"AUC-ROC={best_metrics['auc_roc']:.4f}, \"\n",
        "        f\"Recall={best_metrics['recall']:.4f}, \"\n",
        "        f\"F1={best_metrics['f1']:.4f}. \"\n",
        "    )\n",
        "    if qualified:\n",
        "        others = [n for n in qualified if n != best_name]\n",
        "        if others:\n",
        "            justification += (\n",
        "                f\"Also qualified: {', '.join(others)}. \"\n",
        "                f\"'{best_name}' chosen for highest AUC-ROC among qualifiers.\"\n",
        "            )\n",
        "        else:\n",
        "            justification += \"Only model to pass all three gates.\"\n",
        "    else:\n",
        "        justification += \"No model passed all gates; selected by highest AUC-ROC as fallback.\"\n",
        "\n",
        "    print(f'\\n[select_model] ── RESULT ──')\n",
        "    print(f'  Model         : {best_name}')\n",
        "    print(f'  Justification : {justification}')\n",
        "\n",
        "    return {\n",
        "        **state,\n",
        "        'selected_model':           best_model,\n",
        "        'selected_model_name':      best_name,\n",
        "        'selection_justification':  justification,\n",
        "    }"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "id": "select_model_verify",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "select_model_verify",
        "outputId": "07db4b97-18a8-4d4c-c452-81705e3f10af"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[select_model] ── Selection Criteria ──\n",
            "  AUC-ROC  >= 0.78\n",
            "  Recall   >= 0.7\n",
            "  F1-score >= 0.15\n",
            "\n",
            "[select_model] logistic_regression:\n",
            "  AUC-ROC  = 0.7824  PASS\n",
            "  Recall   = 0.7914  PASS\n",
            "  F1-score = 0.2277  PASS\n",
            "  -> QUALIFIED\n",
            "\n",
            "[select_model] random_forest:\n",
            "  AUC-ROC  = 0.8378  PASS\n",
            "  Recall   = 0.0072  FAIL\n",
            "  F1-score = 0.0141  FAIL\n",
            "  -> ELIMINATED\n",
            "\n",
            "[select_model] xgboost:\n",
            "  AUC-ROC  = 0.8565  PASS\n",
            "  Recall   = 0.0935  FAIL\n",
            "  F1-score = 0.1529  PASS\n",
            "  -> ELIMINATED\n",
            "\n",
            "[select_model] ── RESULT ──\n",
            "  Model         : logistic_regression\n",
            "  Justification : Selected 'logistic_regression' — AUC-ROC=0.7824, Recall=0.7914, F1=0.2277. Only model to pass all three gates.\n",
            "\n",
            "── Verification ──────────────────────────────────────\n",
            "Selected Model : logistic_regression\n",
            "Model Type     : LogisticRegression\n",
            "Justification  : Selected 'logistic_regression' — AUC-ROC=0.7824, Recall=0.7914, F1=0.2277. Only model to pass all three gates.\n"
          ]
        }
      ],
      "source": [
        "result_state = select_model(result_state)\n",
        "\n",
        "print()\n",
        "print('── Verification ──────────────────────────────────────')\n",
        "print(f'Selected Model : {result_state[\"selected_model_name\"]}')\n",
        "print(f'Model Type     : {type(result_state[\"selected_model\"]).__name__}')\n",
        "print(f'Justification  : {result_state[\"selection_justification\"]}')"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "59612cd1",
      "metadata": {
        "id": "59612cd1"
      },
      "source": [
        "# Section 5 — Run Inference Agent Node (CX)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def run_inference(state: AgentState) -> AgentState:\n",
        "    \"\"\"\n",
        "    LangGraph node: Run inference using selected model.\n",
        "\n",
        "    Uses:\n",
        "    - selected_model from state\n",
        "    - X_test as fallback input\n",
        "\n",
        "    Outputs:\n",
        "    - prediction\n",
        "    - fraud_probability\n",
        "    - confidence\n",
        "    - risk_level\n",
        "    \"\"\"\n",
        "\n",
        "    print(\"\\n[run_inference] Running inference...\")\n",
        "\n",
        "    # ── Step 1: Load Data ──────────────────────────────────────────\n",
        "    if \"selected_model\" not in state:\n",
        "        raise ValueError(\"No selected model found in state.\")\n",
        "\n",
        "    model = state[\"selected_model\"]\n",
        "\n",
        "    # ── Step 2: Get Input Data ──────────────────────────────────────────\n",
        "    if \"input_data\" in state:\n",
        "        X_input = state[\"input_data\"]\n",
        "        print(\"[run_inference] Using input_data from state.\")\n",
        "    else:\n",
        "        X_input = state[\"X_test\"].iloc[[0]]\n",
        "        print(\"[run_inference] Using fallback sample from X_test.\")\n",
        "\n",
        "    # ── Step 3: Predict Probability ──────────────────────────────────────────\n",
        "    proba = model.predict_proba(X_input)[0][1]\n",
        "\n",
        "    # ── Step 4: Apply threshold ──────────────────────────────────────────\n",
        "    threshold = 0.3   # Increase recall\n",
        "    prediction = int(proba >= threshold)\n",
        "\n",
        "    # ── Step 5: Confidence ──────────────────────────────────────────\n",
        "    confidence = abs(proba - 0.5) * 2\n",
        "\n",
        "    # ── Step 6: Risk Level ──────────────────────────────────────────\n",
        "    if proba < 0.3:\n",
        "        risk_level = \"Low\"\n",
        "    elif proba < 0.7:\n",
        "        risk_level = \"Medium\"\n",
        "    else:\n",
        "        risk_level = \"High\"\n",
        "\n",
        "    # ── Step 7: Logging ──────────────────────────────────────────\n",
        "    print(f\"[run_inference] Probability: {proba:.4f}\")\n",
        "    print(f\"[run_inference] Prediction: {prediction}\")\n",
        "    print(f\"[run_inference] Confidence: {confidence:.4f}\")\n",
        "    print(f\"[run_inference] Risk Level: {risk_level}\")\n",
        "\n",
        "    # ── Step 8: Return Updated ──────────────────────────────────────────\n",
        "    return {\n",
        "        **state,\n",
        "        \"prediction\": prediction,\n",
        "        \"fraud_probability\": float(proba),\n",
        "        \"confidence\": float(confidence),\n",
        "        \"risk_level\": risk_level\n",
        "    }"
      ],
      "metadata": {
        "id": "n0TfsL2bB3DI"
      },
      "id": "n0TfsL2bB3DI",
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "result_state = run_inference(result_state)\n",
        "\n",
        "print()\n",
        "print('── Verification ──────────────────────────────────────')\n",
        "print(f'Prediction        : {result_state[\"prediction\"]}')\n",
        "print(f'Fraud Probability: {result_state[\"fraud_probability\"]:.4f}')\n",
        "print(f'Confidence       : {result_state[\"confidence\"]:.4f}')\n",
        "print(f'Risk Level       : {result_state[\"risk_level\"]}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vGIGL_98B4N8",
        "outputId": "cd9c981d-8750-4b26-e6e2-aa3ae1959033"
      },
      "id": "vGIGL_98B4N8",
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "[run_inference] Running inference...\n",
            "[run_inference] Using fallback sample from X_test.\n",
            "[run_inference] Probability: 0.6221\n",
            "[run_inference] Prediction: 1\n",
            "[run_inference] Confidence: 0.2442\n",
            "[run_inference] Risk Level: Medium\n",
            "\n",
            "── Verification ──────────────────────────────────────\n",
            "Prediction        : 1\n",
            "Fraud Probability: 0.6221\n",
            "Confidence       : 0.2442\n",
            "Risk Level       : Medium\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "id": "4bee6aab",
      "metadata": {
        "id": "4bee6aab"
      },
      "source": [
        "# Section 6 — Fraud Detection Engine Agent Node(CX)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "id": "2e254e85",
      "metadata": {
        "id": "2e254e85"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import json\n",
        "from typing import Any\n",
        "\n",
        "try:\n",
        "    from openai import OpenAI\n",
        "except Exception:\n",
        "    OpenAI = None\n",
        "\n",
        "\n",
        "def _compute_risk_level(prob: float) -> str:\n",
        "    if prob < 0.30:\n",
        "        return \"Low\"\n",
        "    if prob < 0.70:\n",
        "        return \"Medium\"\n",
        "    return \"High\"\n",
        "\n",
        "\n",
        "def _extract_claim_snapshot(state: AgentState) -> dict:\n",
        "    \"\"\"Prefer raw claim fields; fallback to first row if provided as DataFrame.\"\"\"\n",
        "    if isinstance(state.get(\"input_claim\"), dict):\n",
        "        return state[\"input_claim\"]\n",
        "\n",
        "    if \"new_claim_data\" in state:\n",
        "        data = state[\"new_claim_data\"]\n",
        "        if hasattr(data, \"iloc\"):\n",
        "            return data.iloc[0].to_dict()\n",
        "        if isinstance(data, dict):\n",
        "            return data\n",
        "\n",
        "    # Fallback (may already be encoded features)\n",
        "    if \"input_data\" in state and hasattr(state[\"input_data\"], \"iloc\"):\n",
        "        return state[\"input_data\"].iloc[0].to_dict()\n",
        "\n",
        "    return {}\n",
        "\n",
        "\n",
        "def _rule_reasons(prob: float, risk_level: str, claim: dict) -> list[str]:\n",
        "    reasons = [f\"Model fraud probability is {prob:.2%} ({risk_level} risk band).\"]\n",
        "\n",
        "    def v(key: str):\n",
        "        return claim.get(key)\n",
        "\n",
        "    if v(\"Fault\") == \"Policy Holder\":\n",
        "        reasons.append(\"Claim indicates policy holder is at fault.\")\n",
        "    if v(\"PoliceReportFiled\") == \"No\":\n",
        "        reasons.append(\"No police report was filed for the accident.\")\n",
        "    if v(\"WitnessPresent\") == \"No\":\n",
        "        reasons.append(\"No witness is recorded for the incident.\")\n",
        "    if v(\"PastNumberOfClaims\") in {\"2 to 4\", \"more than 4\"}:\n",
        "        reasons.append(\"Claimant has multiple historical claims.\")\n",
        "    if v(\"NumberOfSuppliments\") in {\"3 to 5\", \"more than 5\"}:\n",
        "        reasons.append(\"High number of supplements is associated with elevated risk.\")\n",
        "    if v(\"AddressChange_Claim\") in {\"under 6 months\", \"1 year\"}:\n",
        "        reasons.append(\"Recent address change before claim submission.\")\n",
        "    if v(\"Days_Policy_Accident\") in {\"none\", \"1 to 7\"}:\n",
        "        reasons.append(\"Very short policy-to-accident interval.\")\n",
        "    if v(\"Days_Policy_Claim\") in {\"none\", \"8 to 15\"}:\n",
        "        reasons.append(\"Short policy-to-claim interval.\")\n",
        "\n",
        "    return reasons\n",
        "\n",
        "\n",
        "def _rule_recommendation(risk_level: str) -> str:\n",
        "    if risk_level == \"High\":\n",
        "        return (\n",
        "            \"Escalate to SIU/manual investigation immediately, pause payout, \"\n",
        "            \"and request full supporting evidence (police report, repair invoices, \"\n",
        "            \"photos, witness/contact statements).\"\n",
        "        )\n",
        "    if risk_level == \"Medium\":\n",
        "        return (\n",
        "            \"Route to senior claims review with targeted document verification \"\n",
        "            \"before approval (policy history, claim timeline, and loss consistency checks).\"\n",
        "        )\n",
        "    return (\n",
        "        \"Proceed with standard processing while keeping lightweight post-payment \"\n",
        "        \"monitoring and random audit sampling.\"\n",
        "    )\n",
        "\n",
        "\n",
        "def _llm_decision(prob: float, risk_level: str, predicted_label: int, reasons: list[str], claim: dict,\n",
        "                  model_name: str = \"gpt-4.1-mini\") -> tuple[str, str]:\n",
        "    if OpenAI is None:\n",
        "        raise RuntimeError(\"openai package not available\")\n",
        "\n",
        "    api_key = os.getenv(\"OPENAI_API_KEY\")\n",
        "    if not api_key:\n",
        "        raise RuntimeError(\"OPENAI_API_KEY is not set\")\n",
        "\n",
        "    client = OpenAI(api_key=api_key)\n",
        "\n",
        "    prompt = {\n",
        "        \"fraud_probability\": round(float(prob), 6),\n",
        "        \"risk_level\": risk_level,\n",
        "        \"predicted_label\": int(predicted_label),\n",
        "        \"rule_signals\": reasons,\n",
        "        \"claim_snapshot\": claim,\n",
        "        \"task\": (\n",
        "            \"Write a concise fraud-review explanation and one actionable recommendation \"\n",
        "            \"for a claims officer. Keep it factual and operational.\"\n",
        "        ),\n",
        "        \"output_schema\": {\n",
        "            \"explanation\": \"string\",\n",
        "            \"recommendation\": \"string\"\n",
        "        }\n",
        "    }\n",
        "\n",
        "    resp = client.chat.completions.create(\n",
        "        model=model_name,\n",
        "        temperature=0.2,\n",
        "        response_format={\"type\": \"json_object\"},\n",
        "        messages=[\n",
        "            {\"role\": \"system\", \"content\": \"You are a fraud-ops assistant for vehicle insurance claims.\"},\n",
        "            {\"role\": \"user\", \"content\": json.dumps(prompt, ensure_ascii=True)}\n",
        "        ]\n",
        "    )\n",
        "\n",
        "    text = resp.choices[0].message.content\n",
        "    payload = json.loads(text)\n",
        "    explanation = str(payload.get(\"explanation\", \"\")).strip()\n",
        "    recommendation = str(payload.get(\"recommendation\", \"\")).strip()\n",
        "\n",
        "    if not explanation or not recommendation:\n",
        "        raise RuntimeError(\"LLM returned empty fields\")\n",
        "\n",
        "    return explanation, recommendation\n",
        "\n",
        "\n",
        "def fraud_detection_engine(state: AgentState,\n",
        "                           use_llm: bool = True,\n",
        "                           llm_model: str = \"gpt-4.1-mini\") -> AgentState:\n",
        "    \"\"\"\n",
        "    LangGraph node: generate fraud decision explanation + recommendation.\n",
        "\n",
        "    Inputs expected from previous nodes:\n",
        "      - fraud_probability\n",
        "      - prediction / predicted_label\n",
        "      - risk_level (optional; auto-computed if missing)\n",
        "      - input_claim or new_claim_data (optional but preferred)\n",
        "\n",
        "    Outputs added to state:\n",
        "      - explanation\n",
        "      - recommendation\n",
        "      - decision_output\n",
        "      - predicted_label (normalized field)\n",
        "      - risk_level (normalized field)\n",
        "    \"\"\"\n",
        "    if \"fraud_probability\" not in state:\n",
        "        raise ValueError(\"fraud_probability missing. Run run_inference first.\")\n",
        "\n",
        "    prob = float(state[\"fraud_probability\"])\n",
        "    risk_level = state.get(\"risk_level\") or _compute_risk_level(prob)\n",
        "\n",
        "    predicted_label = state.get(\"predicted_label\")\n",
        "    if predicted_label is None:\n",
        "        predicted_label = state.get(\"prediction\")\n",
        "    if predicted_label is None:\n",
        "        predicted_label = int(prob >= 0.30)\n",
        "\n",
        "    claim = _extract_claim_snapshot(state)\n",
        "    reasons = _rule_reasons(prob, risk_level, claim)\n",
        "\n",
        "    explanation = \" \".join(reasons)\n",
        "    recommendation = _rule_recommendation(risk_level)\n",
        "    decision_source = \"rule-based\"\n",
        "\n",
        "    if use_llm:\n",
        "        try:\n",
        "            explanation, recommendation = _llm_decision(\n",
        "                prob=prob,\n",
        "                risk_level=risk_level,\n",
        "                predicted_label=int(predicted_label),\n",
        "                reasons=reasons,\n",
        "                claim=claim,\n",
        "                model_name=llm_model,\n",
        "            )\n",
        "            decision_source = f\"llm:{llm_model}\"\n",
        "        except Exception as e:\n",
        "            print(f\"[fraud_detection_engine] LLM unavailable, fallback to rule-based: {e}\")\n",
        "\n",
        "    decision_output = (\n",
        "        f\"Fraud Probability: {prob:.4f}\\n\"\n",
        "        f\"Predicted Label : {int(predicted_label)}\\n\"\n",
        "        f\"Risk Level      : {risk_level}\\n\"\n",
        "        f\"Explanation     : {explanation}\\n\"\n",
        "        f\"Recommendation  : {recommendation}\"\n",
        "    )\n",
        "\n",
        "    print(\"[fraud_detection_engine] Decision generated.\")\n",
        "    print(f\"[fraud_detection_engine] Source: {decision_source}\")\n",
        "\n",
        "    return {\n",
        "        **state,\n",
        "        \"predicted_label\": int(predicted_label),\n",
        "        \"risk_level\": risk_level,\n",
        "        \"explanation\": explanation,\n",
        "        \"recommendation\": recommendation,\n",
        "        \"decision_output\": decision_output,\n",
        "        \"decision_source\": decision_source,\n",
        "    }\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "id": "b85c7479",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b85c7479",
        "outputId": "f62e6f64-411c-408e-f743-3d3ce60c0d61"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[fraud_detection_engine] Decision generated.\n",
            "[fraud_detection_engine] Source: rule-based\n",
            "\n",
            "── Section 6 Verification ───────────────────────────\n",
            "Fraud Probability: 0.6221\n",
            "Predicted Label : 1\n",
            "Risk Level      : Medium\n",
            "Explanation     : Model fraud probability is 62.21% (Medium risk band).\n",
            "Recommendation  : Route to senior claims review with targeted document verification before approval (policy history, claim timeline, and loss consistency checks).\n"
          ]
        }
      ],
      "source": [
        "# Smoke test for Section 6\n",
        "# Default: use_llm=False so notebook runs without API key/network.\n",
        "result_state = fraud_detection_engine(result_state, use_llm=False)\n",
        "\n",
        "print()\n",
        "print(\"── Section 6 Verification ───────────────────────────\")\n",
        "print(result_state[\"decision_output\"])\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "db235256",
      "metadata": {
        "id": "db235256"
      },
      "source": [
        "# Section-7 Langgraph (Om)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "296e29a6",
      "metadata": {
        "id": "296e29a6"
      },
      "source": [
        "# Section-8 - End-to-End Pipeline"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "4216c1fb",
      "metadata": {
        "id": "4216c1fb"
      },
      "source": [
        "# Section 9 - Gradio"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "c4b8dda4",
      "metadata": {
        "id": "c4b8dda4"
      },
      "source": [
        "# Section 10 - 3 test cases"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "1eba1a1a",
      "metadata": {
        "id": "1eba1a1a"
      },
      "source": []
    }
  ],
  "metadata": {
    "jupytext": {
      "cell_metadata_filter": "-all",
      "main_language": "python",
      "notebook_metadata_filter": "-all"
    },
    "kernelspec": {
      "display_name": "base",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.7"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}