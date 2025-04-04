# 🩺 Track_ur_Health 

An AI-powered disease prediction system that analyzes symptoms to provide potential diagnoses, treatment options, and medical recommendations.


## ✨ Features

- **Symptom Analysis**: Advanced parsing and processing of user-reported symptoms
- **Predictive Diagnostics**: AI-based prediction of potential diseases based on symptom patterns
- **Treatment Recommendations**: Suggested treatment options based on predicted conditions
- **Specialist Referrals**: Recommendations for appropriate medical specialists
- **Risk Assessment**: Evaluation of condition severity and risk levels
- **Preventive Measures**: Customized prevention tips for identified conditions

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Track_ur_Health.git
   cd Track_ur_Health
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Run the web interface

```bash
python main.py
```

The Gradio interface will be available at link generated

You'll see an interactive interface where you can:
- Enter symptoms separated by commas
- Adjust the number of results you want to see
- Get predictions along with medical recommendations

## 📊 Dataset Format

The repository includes a `dataset.csv` file ready to use. If you want to use your own dataset, ensure it has the following columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| disease | Name of the medical condition | Influenza |
| symptoms | Comma-separated list of symptoms | fever, cough, sore throat, runny nose |
| cures | Comma-separated list of treatments | rest, fluids, antiviral medication |
| doctor | Medical specialists to consult | General Practitioner, Infectious Disease Specialist |
| risk level | Severity of the condition | moderate |

## 🧠 Model Details

### Machine Learning Pipeline
- **Preprocessing**: Advanced text cleaning, lemmatization, and medical term preservation
- **Data Augmentation**: Oversampling with variations to improve generalization
- **Feature Engineering**: TF-IDF vectorization with n-grams and symptom-disease co-occurrence weighting
- **Model**: Optimized Random Forest classifier with class balancing
- **Performance**: Evaluated using stratified cross-validation

### Key Technical Features
- Automatic model caching for improved performance
- Advanced symptom parsing and normalization
- Disease grouping for improved prediction accuracy
- Risk level assessment
- Preventive measure generation

### Pull Request Process

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📸 Screenshots


![Image](https://github.com/user-attachments/assets/ca941a38-2054-4372-b25d-282223862418)

![Image](https://github.com/user-attachments/assets/33b5502d-e68e-46c0-95a4-ac4cee1c719a)

![Image](https://github.com/user-attachments/assets/8753e16b-7dfd-4602-bc29-b51d48439b53)

![Image](https://github.com/user-attachments/assets/a3b8e904-850c-4a14-82dd-0b5c4a651552)

---

<p align="center">
  <i>Made with ❤️ for better healthcare</i>
</p>
