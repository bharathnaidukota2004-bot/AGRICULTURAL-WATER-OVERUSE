# Mitigating Agricultural Water Overuse via Nash Equilibrium Optimization

**Author:** Bharath Naidu Kota  

---

## 📌 Project Description
Agricultural water overuse and groundwater depletion pose critical threats to environmental and agricultural sustainability in India. This project presents an intelligent decision-support system that models farm water consumption and predicts whether a farm is operating under **Optimal Usage**, **Overuse**, or **Underuse**. 

The system couples **Machine Learning classification models** (Scikit-Learn, Random Forest, Gradient Boosting, Logistic Regression) with **Game-Theoretic Nash Equilibrium optimization**, balancing the marginal utility of meeting crop water requirements against the economic and ecological costs of excessive electric pump operation.

---

## 🚀 Key Features
- **Predictive Water Demand Modeling:** Classifies farm water status using multi-modal agro-climatic telemetry (soil moisture, rainfall, groundwater levels, electricity usage, crop requirements).
- **Nash Equilibrium Optimization:** Integrates an explicit game-theoretic balance score that optimizes water allocation between competing farm agents sharing a common aquifer.
- **Multi-Model Benchmark Studio:** Trains and compares 18 machine learning classifiers with interactive visualization of accuracy, multi-class ROC curves, and confusion matrices.
- **Interactive Web Application:** Built with Flask, featuring responsive form submission, real-time prediction results, and explanation cards.

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Web Framework:** Flask
- **Data Analysis & Processing:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, etc.), Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API), Jinja2 Templating

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/bharathnaidukota2004-bot/AGRICULTURAL-WATER-OVERUSE.git
cd AGRICULTURAL-WATER-OVERUSE
```

### 2. Create and Activate a Virtual Environment (Optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set Environment Variables
Copy `.env.example` to `.env` and set your secret key:
```bash
cp .env.example .env
```

### 5. Run the Application
```bash
python app.py
```

### 6. Open in Browser
Navigate to:
```text
http://127.0.0.1:5000
```

---

## 📊 Results

- **Model Benchmark Accuracy:** 96.67% (Logistic Regression), 96.00% (Gradient Boosting), 95.33% (Random Forest Classifier).
- **Model Accuracy / R²:** `[Insert Model Accuracy / R² Here: e.g., 95.33% Accuracy on 20% Stratified Test Set]`
- **Evaluation Metrics:** Weighted Precision: 95.89%, Weighted Recall: 96.00%, Weighted F1-Score: 95.90%.

---

## 📸 Screenshots

*(Add your screenshots here after running the app)*

1. **Home / Dashboard:**  
   `![Dashboard](static/screenshot_dashboard.png)`
2. **Farm Data Input Form:**  
   `![Form](static/screenshot_form.png)`
3. **Prediction Results & ROC Analysis:**  
   `![Results](static/screenshot_results.png)`

---

## 👨‍💻 Author
**Bharath Naidu Kota**  
*Repository:* [AGRICULTURAL-WATER-OVERUSE](https://github.com/bharathnaidukota2004-bot/AGRICULTURAL-WATER-OVERUSE)
