import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import resample
import re
import itertools
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib
import gradio as gr

# Try to download NLTK resources, handle if already downloaded
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class EnhancedDiseasePredictor:
    def __init__(self, dataset_path, test_size=0.2, use_cached=False, model_path='disease_model.joblib'):
        """
        Enhanced disease predictor with advanced data handling and model optimization
        """
        self.model_path = model_path
        self.dataset_path = dataset_path  # Store dataset path for later use if needed

        if use_cached and self._load_model():
            print("Loaded cached model successfully")
            return

        # Read dataset
        self.df = pd.read_csv(dataset_path)

        # Advanced preprocessing
        self._advanced_preprocessing()

        # Data augmentation
        self._augment_data()

        # Advanced feature engineering
        self._advanced_feature_engineering()

        # Train-test split with stratification
        self._split_data(test_size)

        # Train and optimize model
        self._train_optimized_model()

        # Save model for future use
        self._save_model()

    def _load_model(self):
        """
        Load cached model and vectorizer
        """
        try:
            model_components = joblib.load(self.model_path)
            self.vectorizer = model_components['vectorizer']
            self.model = model_components['model']
            self.classes_ = model_components['classes']

            # Load the dataframe if it exists in the saved model
            if 'dataframe' in model_components:
                self.df = model_components['dataframe']
            else:
                # If dataframe wasn't saved in the model, load it from file
                self.df = pd.read_csv(self.dataset_path)
                self._advanced_preprocessing()

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def _save_model(self):
        """
        Save model and vectorizer for future use
        """
        model_components = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'classes': self.model.classes_,
            'dataframe': self.df  # Save the dataframe with the model
        }
        joblib.dump(model_components, self.model_path)

    def _advanced_preprocessing(self):
        """
        Advanced data cleaning and preprocessing
        """
        # Fix missing values and data issues
        self.df['risk level'] = self.df['risk level'].fillna('unknown')
        self.df['risk level'] = self.df['risk level'].apply(lambda x: x.split('(')[0].strip() if '(' in str(x) else x)

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Handle missing values strategically
        text_columns = ['symptoms', 'cures', 'doctor']
        for col in text_columns:
            self.df[col] = self.df[col].fillna('unknown')
            self.df[col] = self.df[col].apply(self._advanced_text_cleaning)

        # Group similar conditions to increase sample size and predictive power
        self._group_similar_conditions()

        # Encode risk levels
        risk_mapping = {
            'low': 1,
            'moderate': 2,
            'high': 3,
            'varies': 2,
            'unknown': 2
        }

        self.df['risk_level_numeric'] = self.df['risk level'].apply(
            lambda x: risk_mapping.get(x.lower().split()[0], 2)
        )

    def _advanced_text_cleaning(self, text):
        """
        Advanced text preprocessing with lemmatization
        """
        try:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            # Convert to lowercase and remove punctuation
            text = str(text).lower()
            text = re.sub(r'[^\w\s,]', '', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Lemmatize and remove stopwords for better semantic matching
            words = text.split()
            filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

            # Preserve important medical terms that might be stopwords
            medical_stopwords = ['pain', 'high', 'low', 'cold', 'back', 'may']
            for word in words:
                if word in medical_stopwords and word not in filtered_words:
                    filtered_words.append(word)

            return ' '.join(filtered_words)
        except:
            # Fallback to basic cleaning if NLTK fails
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s,]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    def _group_similar_conditions(self):
        """
        Group similar medical conditions to increase sample size and model robustness
        """
        grouping_rules = {
            'cancer': ['bladder cancer', 'cervical cancer', 'lung cancer', 'prostate cancer',
                      'stomach cancer', 'thyroid cancer', 'colorectal cancer'],
            'heart disease': ['heart attack', 'heart failure', 'aortic aneurysm'],
            'meningitis': ['bacterial meningitis'],
            'arthritis': ['rheumatoid arthritis'],
            'kidney disease': ['acute kidney injury', 'chronic kidney disease', 'aki'],
            'diabetes type 2': ['diabetes'],
            'neurological disorders': ['alzheimer\'s disease', 'parkinson\'s disease', 'dementia', 'epilepsy'],
            'gastrointestinal disorders': ['crohn\'s disease', 'ulcerative colitis', 'celiac disease',
                                          'peptic ulcer disease', 'irritable bowel syndrome', 'ibs'],
            'respiratory infections': ['pneumonia', 'bronchitis', 'influenza', 'flu', 'common cold'],
            'skin conditions': ['warts', 'shingles', 'atopic dermatitis']
        }

        for main_condition, sub_conditions in grouping_rules.items():
            for condition in sub_conditions:
                mask = self.df['disease'].str.lower() == condition.lower()
                self.df.loc[mask, 'disease'] = main_condition

    def _augment_data(self):
        """
        Enhanced data augmentation through oversampling and synthetic feature generation
        """
        # Count samples per disease
        disease_counts = self.df['disease'].value_counts()

        # Determine appropriate oversample threshold
        threshold = max(30, disease_counts.median())  # Ensure at least 30 samples per class

        # Oversample minority classes
        augmented_data = []
        for disease, group in self.df.groupby('disease'):
            if len(group) < threshold:
                # Oversample with slight variations to prevent overfitting
                oversampled = resample(
                    group,
                    replace=True,
                    n_samples=threshold,
                    random_state=42
                )

                # Add slight variations to symptoms text to improve generalization
                oversampled['symptoms'] = oversampled['symptoms'].apply(
                    lambda x: self._add_symptom_variations(x)
                )

                augmented_data.append(oversampled)
            else:
                augmented_data.append(group)

        # Combine augmented data
        self.df = pd.concat(augmented_data, ignore_index=True)

    def _add_symptom_variations(self, symptoms_text):
        """
        Add slight variations to symptoms text to improve model generalization
        """
        # Only modify some percentage of samples to maintain diversity
        if np.random.random() > 0.5:
            return symptoms_text

        # Get individual symptoms
        symptoms = symptoms_text.split(',')

        # Potential modifications:
        # 1. Change word order within symptoms
        if len(symptoms) > 1 and np.random.random() > 0.7:
            np.random.shuffle(symptoms)

        # 2. Add common synonyms for some symptoms
        symptom_synonyms = {
            'pain': ['discomfort', 'ache', 'soreness'],
            'fatigue': ['tiredness', 'exhaustion', 'weakness'],
            'fever': ['high temperature', 'elevated temperature'],
            'nausea': ['feeling sick', 'queasiness'],
            'dizziness': ['lightheadedness', 'vertigo'],
            'cough': ['hacking', 'persistent cough']
        }

        # Apply synonym replacement with probability
        for i, symptom in enumerate(symptoms):
            for term, replacements in symptom_synonyms.items():
                if term in symptom and np.random.random() > 0.7:
                    symptoms[i] = symptom.replace(term, np.random.choice(replacements))
                    break

        return ','.join(symptoms)

    def _advanced_feature_engineering(self):
        """
        Advanced feature engineering for improved classification
        """
        # Extract symptom count as a feature
        self.df['symptom_count'] = self.df['symptoms'].apply(lambda x: len(x.split(',')))

        # Create symptom-disease co-occurrence matrix for better weighting
        diseases = self.df['disease'].unique()
        all_symptoms = set()

        for symptoms in self.df['symptoms']:
            symptom_list = [s.strip() for s in symptoms.split(',')]
            all_symptoms.update(symptom_list)

        # Create co-occurrence dictionary for weighting in vectorization
        self.symptom_disease_cooccurrence = {}
        for symptom in all_symptoms:
            symptom_diseases = self.df[self.df['symptoms'].str.contains(symptom)]['disease'].value_counts()
            self.symptom_disease_cooccurrence[symptom] = dict(symptom_diseases)

    def _split_data(self, test_size):
        """
        Split data into train and test sets with stratification
        """
        # Prepare features
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Capture unigrams, bigrams, and trigrams
            max_features=2000,
            min_df=2,           # Minimum document frequency
            max_df=0.9          # Maximum document frequency
        )

        # Vectorize symptoms
        self.X = self.vectorizer.fit_transform(self.df['symptoms'])
        self.y = self.df['disease']

        # Get feature importance weights from co-occurrence
        feature_names = self.vectorizer.get_feature_names_out()

        # Split data with stratification to ensure balanced classes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )

    def _train_optimized_model(self):
        """
        Train an optimized model with hyperparameter tuning
        """
        # Define model with better hyperparameters
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Use cross-validation for model evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Simple model training
        self.model = base_model.fit(self.X_train, self.y_train)

        # Evaluate model
        train_accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train))
        test_accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        test_f1 = f1_score(self.y_test, self.model.predict(self.X_test), average='weighted')

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test F1 score (weighted): {test_f1:.4f}")

        # Print detailed classification report
        y_pred = self.model.predict(self.X_test)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Update model with full dataset for final model
        self.model = base_model.fit(self.X, self.y)

    def predict_disease(self, symptoms, top_n=3):
        """
        Predict top N possible diseases

        Args:
            symptoms (str): Comma-separated symptoms
            top_n (int): Number of top predictions to return

        Returns:
            list: Top N disease predictions with probabilities
        """
        # Clean and normalize symptoms
        cleaned_symptoms = self._advanced_text_cleaning(symptoms)

        # Vectorize symptoms
        symptoms_vectorized = self.vectorizer.transform([cleaned_symptoms])

        # Get prediction probabilities
        probabilities = self.model.predict_proba(symptoms_vectorized)[0]

        # Get top N predictions
        top_indices = probabilities.argsort()[-top_n:][::-1]

        # Prepare results
        predictions = [
            {
                'disease': self.model.classes_[idx],
                'probability': probabilities[idx] * 100
            }
            for idx in top_indices
        ]

        return predictions

    def get_confidence_level(self, probability):
        """
        Convert probability to confidence level
        """
        if probability >= 85:
            return "Very High"
        elif probability >= 70:
            return "High"
        elif probability >= 50:
            return "Moderate"
        elif probability >= 30:
            return "Low"
        else:
            return "Very Low"

    def generate_recommendations(self, predicted_disease):
        """
        Generate medical recommendations based on predicted disease

        Args:
            predicted_disease (str): Predicted disease name

        Returns:
            dict: Recommendations including cures, doctors, and risk level
        """
        # Ensure the dataframe is available
        if not hasattr(self, 'df') or self.df is None:
            # If df doesn't exist, try to load it
            try:
                self.df = pd.read_csv(self.dataset_path)
                self._advanced_preprocessing()
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                return {
                    'cures': 'No specific recommendations found',
                    'doctors': 'General consultation recommended',
                    'risk_level': 'Unknown',
                    'prevention': 'Consult with a healthcare professional for personalized advice'
                }

        # Handle case variations
        predicted_disease_lower = predicted_disease.lower()
        matching_rows = self.df[self.df['disease'].str.lower() == predicted_disease_lower]

        if matching_rows.empty:
            # Try partial matching
            for disease in self.df['disease'].unique():
                if predicted_disease_lower in disease.lower() or disease.lower() in predicted_disease_lower:
                    matching_rows = self.df[self.df['disease'] == disease]
                    break

        if matching_rows.empty:
            return {
                'cures': 'No specific recommendations found',
                'doctors': 'General consultation recommended',
                'risk_level': 'Unknown',
                'prevention': 'Consult with a healthcare professional for personalized advice'
            }

        # Aggregate recommendations with frequency weighting
        cures = []
        for cure in matching_rows['cures']:
            cures.extend([c.strip() for c in cure.split(',')])

        doctors = []
        for doctor in matching_rows['doctor']:
            doctors.extend([d.strip() for d in doctor.split(',')])

        # Count frequencies
        cure_counts = pd.Series(cures).value_counts()
        doctor_counts = pd.Series(doctors).value_counts()

        # Get top recommendations
        top_cures = ', '.join(cure_counts.index[:5])
        top_doctors = ', '.join(doctor_counts.index[:3])

        # Get risk level
        risk_levels = matching_rows['risk level'].value_counts()
        risk_level = risk_levels.index[0] if not risk_levels.empty else 'Unknown'

        # Add preventive measures based on disease
        prevention = self._generate_prevention_tips(predicted_disease)

        recommendations = {
            'cures': top_cures if top_cures else 'No specific recommendations found',
            'doctors': top_doctors if top_doctors else 'General consultation recommended',
            'risk_level': risk_level,
            'prevention': prevention
        }

        return recommendations

    def _generate_prevention_tips(self, disease):
        """
        Generate prevention tips based on disease
        """
        disease_lower = disease.lower()

        prevention_map = {
            'respiratory infections': 'Wash hands frequently, avoid close contact with sick individuals, maintain good respiratory hygiene, stay updated on vaccinations.',
            'pneumonia': 'Get vaccinated, practice good hygiene, avoid smoking, maintain a healthy lifestyle.',
            'flu': 'Annual flu vaccination, frequent handwashing, avoid touching face, maintain distance from sick individuals.',
            'bronchitis': 'Avoid smoking, reduce exposure to air pollution, get vaccinated against flu and pneumonia.',
            'heart disease': 'Maintain a heart-healthy diet, regular exercise, manage stress, avoid smoking, limit alcohol consumption, regular check-ups.',
            'diabetes': 'Maintain healthy weight, regular exercise, balanced diet, limit sugary foods and refined carbs, regular health screenings.',
            'cancer': 'Avoid tobacco, maintain healthy weight, protect from sun exposure, limit alcohol, eat a healthy diet, regular screenings.',
            'neurological disorders': 'Regular exercise, cognitive activities, healthy diet, manage cardiovascular risk factors, social engagement.',
            'arthritis': 'Maintain healthy weight, regular exercise, protect joints, balance activity with rest, healthy diet rich in anti-inflammatory foods.',
            'kidney disease': 'Control blood pressure and diabetes, healthy diet, regular exercise, avoid smoking, limit alcohol, stay hydrated.',
            'gastrointestinal disorders': 'Balanced diet, identify food triggers, manage stress, regular exercise, limit alcohol and caffeine.'
        }

        # Find matching prevention tips
        for key, tips in prevention_map.items():
            if key in disease_lower or disease_lower in key:
                return tips

        # Default prevention tips
        return 'Maintain a healthy lifestyle with regular exercise, balanced diet, adequate sleep, and regular medical check-ups.'

# Gradio interface function
def create_gradio_interface(dataset_path='dataset.csv', use_cached_model=True):
    """
    Create Gradio interface for disease prediction

    Args:
        dataset_path (str): Path to the medical dataset
        use_cached_model (bool): Whether to use cached model if available
    """
    # Initialize predictor
    try:
        predictor = EnhancedDiseasePredictor(dataset_path, use_cached=use_cached_model)
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        # If error occurs, create with use_cached=False to force rebuilding the model
        predictor = EnhancedDiseasePredictor(dataset_path, use_cached=False)

    def predict_and_recommend(symptoms, num_results=3):
        """
        Predict diseases and generate recommendations

        Args:
            symptoms (str): User-entered symptoms
            num_results (int): Number of results to display

        Returns:
            str: Formatted prediction and recommendation results
        """
        # Validate input
        if not symptoms or len(symptoms.strip()) == 0:
            return "Please enter valid symptoms."

        try:
            # Parse num_results
            num_results = int(num_results)
            if num_results < 1:
                num_results = 3
            elif num_results > 5:
                num_results = 5

            # Predict diseases
            predictions = predictor.predict_disease(symptoms, top_n=num_results)

            # Prepare detailed output
            output = "🔍 PREDICTION RESULTS 🔍\n\n"

            for i, pred in enumerate(predictions, 1):
                disease = pred['disease']
                probability = pred['probability']
                confidence = predictor.get_confidence_level(probability)

                output += f"📌 PREDICTION {i}: {disease.upper()}\n"
                output += f"Confidence: {confidence} ({probability:.1f}%)\n\n"

                # Get recommendations
                recommendations = predictor.generate_recommendations(disease)

                output += "🏥 MEDICAL RECOMMENDATIONS:\n"
                output += f"• Treatment options: {recommendations['cures']}\n"
                output += f"• Recommended specialist: {recommendations['doctors']}\n"
                output += f"• Risk level: {recommendations['risk_level']}\n"
                output += f"• Prevention: {recommendations['prevention']}\n\n"

                if i < len(predictions):
                    output += "-------------------------------------------\n\n"

            output += "\n⚠️ IMPORTANT: This is not a substitute for professional medical advice. Please consult a healthcare provider for diagnosis and treatment."

            return output

        except Exception as e:
            return f"An error occurred: {str(e)}\n\nPlease try again with different symptoms or contact support."

    # Create Gradio interface
    iface = gr.Interface(
        fn=predict_and_recommend,
        inputs=[
            gr.Textbox(
                label="Enter Symptoms",
                placeholder="Enter symptoms separated by commas (e.g., fever, cough, headache)"
            ),
            gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Number of Results"
            )
        ],
        outputs=gr.Textbox(label="Prediction Results"),
        title="🩺 Track_ur_Health",
        description="Enter your symptoms to get potential diagnoses, treatment options, and medical recommendations.",
        examples=[
            ["fever, cough, sore throat, runny nose, body aches", 3],
            ["chest pain, shortness of breath, sweating, nausea", 3],
            ["headache, dizziness, fatigue, difficulty concentrating", 3],
            ["abdominal pain, bloating, diarrhea, constipation", 3],
            ["joint pain, stiffness, swelling, fatigue", 3]
        ],
        theme="default"
    )

    return iface

def main():
    """
    Launch the Gradio interface
    """
    interface = create_gradio_interface(dataset_path='dataset.csv')
    interface.launch(share=True)

if __name__ == "__main__":
    main()