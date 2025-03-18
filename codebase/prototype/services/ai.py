import os
from http.client import HTTPException
from typing import List

import numpy as np
from django.db.models import QuerySet
from django.forms import model_to_dict
from joblib import load
from sentence_transformers import SentenceTransformer

from prototype.api.schemas import ContributionPrediction, ContributionsSchemaOutput
from prototype.models import Employee
from prototype.models.contributions import Contributions
from prototype.utils.ai import load_yaml_file
from prototype.utils.choices import ContributionStatus
from prototype.utils.management import get_csv_path



class AiService:
    def __init__(self):
        # self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.prompt = load_yaml_file(file_name="contributions-prompt.yaml")

    def call_external_service(self, prompt: str):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _update_prompt_with_details(self, **kwargs):
        prompt = self.prompt
        for key, val in kwargs.items():
            prompt.format(key=val)

        return prompt

    def generate_ai_content(self, employee: Employee, contributions: List[Contributions]) -> str:
        prompt = self._update_prompt_with_details(
            employee_name=f"{employee.first_name} {employee.last_name}",
            employee_address=f"{employee.address}",
            contributions_list=contributions,
        )

        response = self.call_external_service(prompt=prompt)
        return response

    def _predict_false_positives_using_semantic_embedding(
            self,employee: Employee,
            contributions: QuerySet[Contributions],
            show_probability: bool = False,
    ):
        # 3. Load the trained model and scaler
        model_filename = get_csv_path(filename="incremental_model.pkl", extra_path=None)
        scaler_filename = get_csv_path(filename="scaler.pkl", extra_path=None)
        if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            raise HTTPException(status_code=500, detail="Model or scaler not found. Please train the model first.")
        model = load(model_filename)
        scaler = load(scaler_filename)

        # 4. Initialize SentenceTransformer for inference
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        def get_embedding(text):
            if not text or text.strip() == "":
                text = " "
            return st_model.encode(text)

        # 5. Generate employee embeddings
        emp_name_text = f"{employee.first_name} {employee.last_name}"
        emp_addr_text = employee.address or ""
        emp_name_emb = get_embedding(emp_name_text)
        emp_addr_emb = get_embedding(emp_addr_text)

        predictions = []
        # 6. Loop through contributions, generate embeddings, scale, and predict
        for ctr in contributions:
            ctr_name_text = f"{ctr.first_name} {ctr.last_name}"
            ctr_addr_text = ctr.address or ""
            ctr_name_emb = get_embedding(ctr_name_text)
            ctr_addr_emb = get_embedding(ctr_addr_text)

            # Concatenate embeddings in the same order as training
            x_infer = np.concatenate([emp_name_emb, ctr_name_emb, emp_addr_emb, ctr_addr_emb]).reshape(1, -1)
            # Normalize using the same scaler

            x_infer_scaled = scaler.transform(x_infer)

            pred_label = model.predict(x_infer_scaled)[0]
            pred_prob = model.predict_proba(x_infer_scaled)[0][1]

            if show_probability:
                predictions.append(ContributionPrediction(
                    contribution_id=ctr.id,
                    contributor_name=f"{ctr.first_name} {ctr.last_name}",
                    contributor_address=ctr.address,
                    predicted_label=int(pred_label),
                    prob_of_true_match=float(f"{pred_prob:.3f}")
                ))
            else:
                if pred_label == 0:
                    predictions.append(
                        ContributionsSchemaOutput(
                            **{
                                "id": ctr.id,
                                **model_to_dict(
                                    ctr
                                ),
                                "prediction_label": pred_label,
                                "status": ContributionStatus.FALSE_POSITIVE
                            }
                        )

                    )


        return predictions



