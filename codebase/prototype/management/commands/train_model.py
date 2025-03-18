# ml_training/management/commands/train_model.py


from prototype.utils.management import get_csv_path  # or wherever you define get_csv_path
import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sentence_transformers import SentenceTransformer
from prototype.utils.management import get_csv_path

import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from joblib import dump, load
from sklearn.linear_model import SGDClassifier
from sentence_transformers import SentenceTransformer

# If you have a utility function to resolve CSV paths, import it; otherwise, use the path as is.

import os
import numpy as np
import pandas as pd
from joblib import dump, load
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


# import os
# import numpy as np
# import pandas as pd
# from django.core.management.base import BaseCommand
# from joblib import dump, load
# from sklearn.linear_model import SGDClassifier
# from sentence_transformers import SentenceTransformer
from prototype.utils.management import get_csv_path  # Adjust as needed


class Command(BaseCommand):
    help = "Incrementally train/update the employee-contributor matching model using SentenceTransformer embeddings."

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv',
            default='data.csv',
            help='Path to the CSV file containing training data.'
        )
        # Hyperparameter options
        parser.add_argument('--alpha', type=float, default=0.0001, help='Regularization strength (alpha)')
        parser.add_argument('--learning_rate', default='optimal', choices=['constant', 'optimal', 'invscaling', 'adaptive'],
                            help='Learning rate schedule')
        parser.add_argument('--eta0', type=float, default=0.0, help='Initial learning rate when learning_rate is constant')

    def handle(self, *args, **options):
        # ----- 1. Initialize SentenceTransformer Model -----
        self.stdout.write(self.style.NOTICE("Loading SentenceTransformer model..."))
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.stdout.write(self.style.SUCCESS("SentenceTransformer model loaded."))

        # ----- 2. Helper to get embeddings -----
        def get_embedding(text):
            if not text or text.strip() == "":
                text = " "
            return st_model.encode(text)

        # ----- 3. Read and Validate CSV Data -----
        csv_path = options['csv']
        csv_path = get_csv_path(csv_path)
        self.stdout.write(self.style.NOTICE(f"Reading data from {csv_path}..."))
        df = pd.read_csv(csv_path)

        required_cols = [
            "employee_first_name", "employee_last_name", "employee_address",
            "contributor_first_name", "contributor_last_name", "contributor_address", "label"
        ]
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].fillna("")
            else:
                raise ValueError(f"Required column '{col}' not found in CSV.")

        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        if df["label"].isna().any():
            raise ValueError("Some labels are missing or non-numeric.")

        # ----- 4. Generate Embeddings for Each Row -----
        self.stdout.write(self.style.NOTICE("Generating embeddings using SentenceTransformer..."))
        emp_name_embeds = []
        ctr_name_embeds = []
        emp_addr_embeds = []
        ctr_addr_embeds = []

        for _, row in df.iterrows():
            emp_name_text = f"{row['employee_first_name']} {row['employee_last_name']}".strip()
            ctr_name_text = f"{row['contributor_first_name']} {row['contributor_last_name']}".strip()

            emp_name_embeds.append(get_embedding(emp_name_text))
            ctr_name_embeds.append(get_embedding(ctr_name_text))
            emp_addr_embeds.append(get_embedding(row["employee_address"]))
            ctr_addr_embeds.append(get_embedding(row["contributor_address"]))

        emp_name_embeds = np.array(emp_name_embeds)
        ctr_name_embeds = np.array(ctr_name_embeds)
        emp_addr_embeds = np.array(emp_addr_embeds)
        ctr_addr_embeds = np.array(ctr_addr_embeds)

        # ----- 5. Combine Embeddings into a Feature Matrix -----
        self.stdout.write(self.style.NOTICE("Combining embeddings into a feature matrix..."))
        X = np.concatenate([emp_name_embeds, ctr_name_embeds, emp_addr_embeds, ctr_addr_embeds], axis=1)
        y = df["label"].values.astype(int)
        self.stdout.write(self.style.NOTICE(f"Feature matrix shape: {X.shape}"))


        # ----- 6. Normalize Embeddings with StandardScaler -----
        scaler_filename = "scaler.pkl"
        if os.path.exists(scaler_filename):
            scaler = load(scaler_filename)
            # Update the scaler with new data
            scaler.partial_fit(X)
            X_scaled = scaler.transform(X)
            self.stdout.write(self.style.NOTICE("Updated existing scaler with new data."))
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.stdout.write(self.style.NOTICE("Fitted new scaler on training data."))

        # ----- 7. Load or Initialize the Incremental Model with Hyperparameters -----
        model_filename = "incremental_model.pkl"
        if os.path.exists(model_filename):
            model = load(model_filename)
            self.stdout.write(self.style.NOTICE("Loaded existing incremental model."))
        else:
            model = SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                tol=1e-3,
                alpha=options['alpha'],
                learning_rate=options['learning_rate'],
                eta0=options['eta0']
            )
            self.stdout.write(self.style.NOTICE("Initialized new incremental model with hyperparameters."))

        # ----- 8. Incremental Training with partial_fit -----
        classes = np.array([0, 1])
        model.partial_fit(X_scaled, y, classes=classes)
        self.stdout.write(self.style.SUCCESS("Model updated with partial_fit on current data."))

        training_acc = model.score(X_scaled, y)
        self.stdout.write(self.style.SUCCESS(f"Training accuracy on current batch: {training_acc:.4f}"))

        # ----- 9. Save the Updated Model and Scaler -----
        dump(model, model_filename)
        dump(scaler, scaler_filename)
        self.stdout.write(self.style.SUCCESS(f"Incremental model saved to {model_filename} and scaler saved to {scaler_filename}."))




# class Command(BaseCommand):
#     help = "Incrementally train/update the employee-contributor matching model using SentenceTransformer embeddings."
#
#     def add_arguments(self, parser):
#         parser.add_argument(
#             '--csv',
#             default='data.csv',
#             help='Path to the CSV file containing training data.'
#         )
#
#     def handle(self, *args, **options):
#         # ----- 1. Initialize SentenceTransformer Model -----
#         self.stdout.write(self.style.NOTICE("Loading SentenceTransformer model..."))
#         st_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.stdout.write(self.style.SUCCESS("SentenceTransformer model loaded."))
#
#         # ----- 2. Define helper function to get embeddings using SentenceTransformer -----
#         def get_embedding(text):
#             if not text or text.strip() == "":
#                 text = " "
#             return st_model.encode(text)
#
#         # ----- 3. Read and Validate CSV Data -----
#         csv_path = options['csv']
#         csv_path = get_csv_path(csv_path)
#         self.stdout.write(self.style.NOTICE(f"Reading data from {csv_path}..."))
#         df = pd.read_csv(csv_path)
#
#         required_cols = [
#             "employee_first_name", "employee_last_name", "employee_address",
#             "contributor_first_name", "contributor_last_name", "contributor_address", "label"
#         ]
#         for col in required_cols:
#             if col in df.columns:
#                 df[col] = df[col].fillna("")
#             else:
#                 raise ValueError(f"Required column '{col}' not found in CSV.")
#
#         df["label"] = pd.to_numeric(df["label"], errors="coerce")
#         if df["label"].isna().any():
#             raise ValueError("Some labels are missing or non-numeric.")
#
#         # ----- 4. Generate Embeddings for Each Row -----
#         self.stdout.write(self.style.NOTICE("Generating embeddings using SentenceTransformer..."))
#         emp_name_embeds = []
#         ctr_name_embeds = []
#         emp_addr_embeds = []
#         ctr_addr_embeds = []
#
#         for _, row in df.iterrows():
#             emp_name_text = f"{row['employee_first_name']} {row['employee_last_name']}".strip()
#             ctr_name_text = f"{row['contributor_first_name']} {row['contributor_last_name']}".strip()
#
#             emp_name_embeds.append(get_embedding(emp_name_text))
#             ctr_name_embeds.append(get_embedding(ctr_name_text))
#             emp_addr_embeds.append(get_embedding(row["employee_address"]))
#             ctr_addr_embeds.append(get_embedding(row["contributor_address"]))
#
#         emp_name_embeds = np.array(emp_name_embeds)
#         ctr_name_embeds = np.array(ctr_name_embeds)
#         emp_addr_embeds = np.array(emp_addr_embeds)
#         ctr_addr_embeds = np.array(ctr_addr_embeds)
#
#         # ----- 5. Combine Embeddings into a Feature Matrix -----
#         # "all-MiniLM-L6-v2" outputs 384-dim vectors; concatenating 4 fields gives 1536-dim vector.
#         self.stdout.write(self.style.NOTICE("Combining embeddings into a feature matrix..."))
#         X = np.concatenate([emp_name_embeds, ctr_name_embeds, emp_addr_embeds, ctr_addr_embeds], axis=1)
#         y = df["label"].values.astype(int)
#         self.stdout.write(self.style.NOTICE(f"Feature matrix shape: {X.shape}"))
#
#         # ----- 6. Load or Initialize the Incremental Model -----
#         model_filename = "incremental_model.pkl"
#         if os.path.exists(model_filename):
#             model = load(model_filename)
#             self.stdout.write(self.style.NOTICE("Loaded existing incremental model."))
#         else:
#             model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
#             self.stdout.write(self.style.NOTICE("Initialized new incremental model."))
#
#         # ----- 7. Incremental Training with partial_fit -----
#         classes = np.array([0, 1])
#         model.partial_fit(X, y, classes=classes)
#         self.stdout.write(self.style.SUCCESS("Model updated with partial_fit on current data."))
#
#         # Optionally, evaluate training accuracy on the current batch
#         training_acc = model.score(X, y)
#         self.stdout.write(self.style.SUCCESS(f"Training accuracy on current batch: {training_acc:.4f}"))
#
#         # ----- 8. Save the Updated Model -----
#         dump(model, model_filename)
#         self.stdout.write(self.style.SUCCESS(f"Incremental model saved to {model_filename}."))
#
#         # Now, each time you run this command with new CSV data,
#         # the model is updated incrementally and retains previous training.

# class Command(BaseCommand):
#     help = "Incrementally train/update the employee-contributor matching model using SentenceTransformer embeddings."
#
#     def add_arguments(self, parser):
#         parser.add_argument(
#             '--csv',
#             default='data.csv',
#             help='Path to the CSV file containing training data.'
#         )
#
#     def handle(self, *args, **options):
#         # ----- 1. Initialize SentenceTransformer Model -----
#         self.stdout.write(self.style.NOTICE("Loading SentenceTransformer model..."))
#         st_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.stdout.write(self.style.SUCCESS("SentenceTransformer model loaded."))
#
#         # ----- 2. Helper function to get embeddings -----
#         def get_embedding(text):
#             if not text or text.strip() == "":
#                 text = " "
#             return st_model.encode(text)
#
#         # ----- 3. Read CSV Data -----
#         csv_path = options['csv']
#         csv_path = get_csv_path(csv_path)
#         self.stdout.write(self.style.NOTICE(f"Reading data from {csv_path}..."))
#         df = pd.read_csv(csv_path)
#
#         required_cols = [
#             "employee_first_name", "employee_last_name", "employee_address",
#             "contributor_first_name", "contributor_last_name", "contributor_address", "label"
#         ]
#         for col in required_cols:
#             if col in df.columns:
#                 df[col] = df[col].fillna("")
#             else:
#                 raise ValueError(f"Required column '{col}' not found in CSV.")
#         df["label"] = pd.to_numeric(df["label"], errors="coerce")
#         if df["label"].isna().any():
#             raise ValueError("Some labels are missing or non-numeric.")
#
#         # ----- 4. Generate Embeddings for Each Row -----
#         self.stdout.write(self.style.NOTICE("Generating embeddings..."))
#         emp_name_embeds = []
#         ctr_name_embeds = []
#         emp_addr_embeds = []
#         ctr_addr_embeds = []
#         for _, row in df.iterrows():
#             emp_name_text = f"{row['employee_first_name']} {row['employee_last_name']}".strip()
#             ctr_name_text = f"{row['contributor_first_name']} {row['contributor_last_name']}".strip()
#             emp_name_embeds.append(get_embedding(emp_name_text))
#             ctr_name_embeds.append(get_embedding(ctr_name_text))
#             emp_addr_embeds.append(get_embedding(row["employee_address"]))
#             ctr_addr_embeds.append(get_embedding(row["contributor_address"]))
#
#         emp_name_embeds = np.array(emp_name_embeds)
#         ctr_name_embeds = np.array(ctr_name_embeds)
#         emp_addr_embeds = np.array(emp_addr_embeds)
#         ctr_addr_embeds = np.array(ctr_addr_embeds)
#
#         # ----- 5. Combine into Feature Matrix -----
#         # For "all-MiniLM-L6-v2", each embedding is 384 dimensions.
#         # Concatenating 4 fields gives a feature vector of dimension 1536.
#         self.stdout.write(self.style.NOTICE("Combining embeddings into a feature matrix..."))
#         X = np.concatenate([emp_name_embeds, ctr_name_embeds, emp_addr_embeds, ctr_addr_embeds], axis=1)
#         y = df["label"].values.astype(int)
#         self.stdout.write(self.style.NOTICE(f"Feature matrix shape: {X.shape}"))
#
#         # ----- 6. (Optional) Cross-validation -----
#         self.stdout.write(self.style.NOTICE("Running 5-fold stratified cross-validation..."))
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         clf_for_cv = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
#         cv_scores = cross_val_score(clf_for_cv, X, y, cv=skf)
#         self.stdout.write(self.style.SUCCESS(f"CV average accuracy: {cv_scores.mean():.4f}"))
#
#         # ----- 7. Train/Test Split and Final Training -----
#         self.stdout.write(self.style.NOTICE("Splitting data into train/test..."))
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#         self.stdout.write(self.style.NOTICE("Training RandomForestClassifier model..."))
#         clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
#         clf.fit(X_train, y_train)
#
#         # ----- 8. Evaluate and Save the Model -----
#         self.stdout.write(self.style.NOTICE("Evaluating model..."))
#         y_pred = clf.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred)
#         self.stdout.write(self.style.SUCCESS(f"Test Accuracy: {acc:.4f}"))
#         self.stdout.write(self.style.SUCCESS("Classification Report:\n" + report))
#
#         model_path = "model.joblib"
#         dump(clf, model_path)
#         self.stdout.write(self.style.SUCCESS(f"Model saved to {model_path}."))
#
#         # For debugging: if you want to start with a small dataset,
#         # try training with a single or few examples first to ensure the pipeline works,
#         # then gradually add more data.
