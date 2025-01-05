import streamlit as st
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

def load_model(run_id, model_name):
    """Carica un modello salvato da MLflow"""
    model_filename = {
        "OLS": "ols_model",
        "Ridge": "ridge_model",
        "Lasso": "lasso_model",
        "RandomForest": "rf_model"
    }
    
    filename = model_filename[model_name]
    model = mlflow.sklearn.load_model(f"mlruns/719251958895265794/{run_id}/artifacts/{filename}")
    return model

def get_latest_run_id(model_type):
    """Recupera l'ID del run più recente per un dato tipo di modello"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("House_Price_Prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_type = '{model_type}'"
    )
    
    if runs:
        return runs[0].info.run_id
    return None

def main():
    st.title("House Price Prediction")
    
    # Sidebar per selezionare il modello
    model_type = st.sidebar.selectbox(
        "Select Model", 
        ["OLS", "Ridge", "Lasso", "RandomForest"]
    )
    
    # Lista ordinata delle features IN ORDINE ALFABETICO come nel modello
    features_order = [
        "company_rating",
        "crew",
        "d_check_complete",
        "engines",
        "iata_approved",
        "moon_clearance_complete",
        "passenger_capacity",
        "review_scores_rating"
    ]
    
    # Input features
    st.header("Input Features")
    feature_values = {}
    
    # Raccogli input mantenendo l'ordine
    for feature in features_order:
        if feature in ['d_check_complete', 'moon_clearance_complete', 'iata_approved']:
            feature_values[feature] = int(st.checkbox(feature.replace('_', ' ').title()))
        elif feature in ['company_rating', 'review_scores_rating']:
            feature_values[feature] = st.number_input(feature.replace('_', ' ').title(), 
                                                    min_value=0, max_value=100)
        else:
            feature_values[feature] = st.number_input(feature.replace('_', ' ').title(), 
                                                    min_value=0)
    
    # Crea DataFrame con input nell'ordine corretto
    input_data = pd.DataFrame([feature_values])
    input_data = input_data[features_order]
    
    if st.button("Predict"):
        run_id = get_latest_run_id(model_type)
        
        if run_id is None:
            st.error(f"No runs found for model type {model_type}")
            return
        
        try:
            model = load_model(run_id, model_type)
            prediction = model.predict(input_data)
            
            # Mostra il risultato
            st.success(f"Predicted Price: ${prediction[0]:,.2f}")
            
            # Mostra metriche del modello
            client = MlflowClient()
            run = client.get_run(run_id)
            st.write("Model Metrics:")
            st.write(f"R2 Score (Test): {run.data.metrics.get('test_r2', 'N/A'):.3f}")
            st.write(f"R2 Score (CV): {run.data.metrics.get('cv_r2_mean', 'N/A'):.3f}")
            
        except Exception as e:
            st.error(f"Error loading/using model: {str(e)}")

if __name__ == "__main__":
    main()
