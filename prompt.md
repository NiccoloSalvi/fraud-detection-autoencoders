**Goal:** Replicate the solution described in the paper **“Combining Autoencoders and Deep Learning for Effective Fraud Detection in Credit Card Transactions.”** The paper proposes:

1. **ASVM Approach** (Autoencoder + SVM)  
   - Use an **autoencoder** to learn how to reconstruct fraudulent transactions, then generate synthetic fraud samples.  
   - Filter these generated samples through a **support vector machine** classifier (trained on an undersampled version of the data) to discard any generated samples that do not match the “fraud profile.”  
   - Augment the minority (fraud) class with these validated synthetic samples until the fraud class count equals that of the legitimate class.

2. **GB_ALSTM Model** (Gradient Boosting with an Attention‑LSTM Base Learner)  
   - Once the training set is balanced via ASVM, fit a **gradient boosting ensemble** where each “weak learner” is an **attention‑based LSTM** model.  
   - Each new ALSTM attempts to predict the residuals of the previous ensemble iteration, continuing until we reach a final boosted model.

3. **Benchmark Models**  
   - The paper also compares other deep learning models (e.g., plain LSTM, ALSTM, CNN, ANN, BLSTM) to show that the GB_ALSTM + ASVM approach outperforms them on the European credit card fraud dataset.

4. **Dataset**  
   - Uses the **European credit card transactions dataset** from Kaggle (284,807 samples, 492 fraud, 284,315 legit).  
   - The “Time” feature is often discarded; “Amount” and the PCA columns (V1–V28) are typically used.  
   - The final label is in the “Class” column: 1 = Fraud, 0 = Legit.

5. **Metrics & Results**  
   - The paper reports **Accuracy, Precision, Recall, F1, and Specificity**, plus confusion matrices.  
   - Their best model claims ~99.99% accuracy, ~97%+ recall, and similarly high precision.

---

## **CORE PIPELINE TO REPLICATE**

1. **Data Preprocessing**  
   - Load the CSV file (commonly `creditcard.csv`).  
   - Drop or keep “Time” (the paper often discards it).  
   - Scale/normalize “Amount” if needed (e.g., via RobustScaler or standardization).  
   - Split into train (70%) and test (30%). Possibly do a further validation split if needed.

2. **Imbalanced Learning with Autoencoder & SVM (ASVM)**  
   1. **Undersample** the majority class for your initial training set to reduce the total data but keep the minority class intact.  
   2. **Train an Autoencoder** *only* on the minority (fraud) subset.  
      - The autoencoder’s architecture in the paper uses multiple dense layers (encoder → bottleneck → decoder).  
      - After training, generate new fraud samples by passing random noise or known fraudulent samples through the autoencoder.  
   3. **Train SVM** on your (undersampled) training set (fraud + legit) to classify fraud vs. legit.  
   4. **Filter** newly generated fraud samples by letting the SVM confirm they’re fraud. Only retain samples that SVM labels as “fraud.”  
   5. **Repeat** until the number of fraud samples ~ matches the number of legitimate samples.

3. **Gradient Boosting + ALSTM**  
   1. **Architecture**: The base learner is an Attention‑LSTM (ALSTM).  
   2. **GB Loop** (per the paper’s Algorithm 2):  
      - Initialize an ensemble \(h_0\) (often a constant).  
      - For \(t=1\) to \(N\_estimators\):  
        - Compute residuals w.r.t. the previous model’s predictions.  
        - Fit a new ALSTM to these residuals.  
        - Find optimal step size \(\alpha_t\).  
        - Update the ensemble.  
   3. **Final Model**: The ensemble predictor is the sum of all ALSTMs, each scaled by \(\alpha_t\).

4. **Evaluation**  
   - On the balanced training set, train the final GB_ALSTM model.  
   - Use the unseen 30% test set for evaluation.  
   - Report confusion matrix, accuracy, precision, recall, specificity, F1.  
   - Compare to plain LSTM, plain ALSTM, CNN, ANN, BLSTM, etc.

5. **Target Performance**  
   - The paper mentions ~99.99% accuracy, ~97% recall, ~98% precision (depending on runs).  
   - Show that the final method outperforms simpler deep learning models.
