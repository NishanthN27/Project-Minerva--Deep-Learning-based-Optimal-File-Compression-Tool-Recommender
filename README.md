# Project Minerva: AI-Powered File Compression Optimizer

Project Minerva is a Streamlit-based web app that uses deep learning models to recommend the best compression tool for any given file. Instead of relying on simple heuristics like file extensions—which are often unreliable—it analyzes the actual content and properties of the file. This content-aware approach makes the system both faster than brute-force testing and more accurate than traditional rule-based methods.

The application allows a user to upload a file, select from one of five different AI models, and receive an instant recommendation. To prove its effectiveness, the app also runs a full benchmark in the background, providing a detailed comparison of compression ratios, final file sizes, and the time saved by using the DL-powered approach.

## Key Features

-   **Intelligent Prediction:** Uses deep learning to analyze file content and predict the best compressor.
-   **Multi-Model Analysis:** Allows users to select from five different, fully tuned AI architectures to compare their recommendations.
-   **Live Benchmarking:** Provides immediate, tangible proof of the model's accuracy by comparing its choice against other standard tools.
-   **Efficiency Dashboard:** Quantifies the time saved by using the ML pipeline versus a brute-force approach.
  
## Models Implemented

The application allows you to choose from five distinct, tuned models, each testing a different architectural hypothesis:

1.  **Baseline MLP:** A fundamental deep learning benchmark.
2.  **Robust MLP:** An industry-standard MLP with Dropout and Batch Normalization to prevent overfitting.
3.  **Wide & Deep Network:** A hybrid architecture that learns both simple rules and complex patterns simultaneously.
4.  **ResNet-style MLP:** A very deep network using residual connections to learn a richer hierarchy of features.
5.  **DL-ML Hybrid:** A state-of-the-art two-stage model using a DNN for feature extraction and an XGBoost classifier for the final prediction (this is the default and best-performing model).

## Local Setup

### Prerequisites

-   Python 3.8 - 3.12
-   Git for cloning the repository.
-   The command-line versions of various compression tools installed and available in your system's PATH.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd project-minerva
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    The `requirements.txt` file contains all the necessary libraries for the project.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Model and Artifact Files:**
    Ensure the `saved_models/` directory is present and contains all the pre-trained models and artifacts. This directory should include:
    -   `my_scaler.gz` (the feature scaler)
    -   `label_encoder.pkl` (the target label encoder)
    -   `tuned_feature_extractor.h5` (for the DL-ML Hybrid model)
    -   All five final model files (e.g., `tuned_robust_mlp_model.h5`, `tuned_xgb_classifier.pkl`, etc.)

5.  **Install Compression Tools**

### Running the Application

Once the setup is complete, run the following command from the root of the project directory:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser, typically at `http://localhost:8501`.

### Usage

1.  **Select an AI Model:** Use the dropdown menu on the left to choose which of the five trained models you want to use for the prediction. The "DL-ML Hybrid" is selected by default as it is the best-performing model.
2.  **Upload a File:** Drag and drop a supported file (`.txt`, `.csv`, `.json`, `.pdf`, `.png`, `.jpg`, `.wav`) onto the upload area.
3.  **Analyze:** Click the "Analyze & Recommend" button to start the process.
4.  **Review Results:**
    -   The AI's recommendation will appear instantly at the top.
    -   The app will then run the full benchmark. Once complete, a dashboard will appear with a comparative bar chart, a detailed results table, and an efficiency analysis.
    -   You can also download the file compressed with the recommended tool.

### Troubleshooting

-   **"Model not found" error:** Ensure that the `saved_models/` directory is in the correct location and contains all the required `.h5`, `.pkl`, and `.gz` files.
-   **"Compression tool not found" warning in `inference.py`:** This means one of the compression tools is not installed or not in your system's PATH. The prediction will still work, but the benchmarking feature will be incomplete.
-   **File Size Limit Error:** The live demo is capped at a 50MB file size to ensure quick response times. Please use a smaller file for analysis.
