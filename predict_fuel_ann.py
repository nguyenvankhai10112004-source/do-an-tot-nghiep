import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mo_hinh_ann_7_features.keras')
SCALER_X_PATH = os.path.join(os.path.dirname(__file__), 'scaler_X_7features.pkl')
SCALER_Y_PATH = os.path.join(os.path.dirname(__file__), 'scaler_y_7features.pkl')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder_trang_thai.pkl')


def load_ann_model():
    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler_X, scaler_y, label_encoder


def predict_fuel(energy_features, quang_duong_km, traffic_status):
    """
    energy_features: dict with keys E_roll_kJ, E_aero_kJ, E_inertia_kJ, E_grade_kJ, E_curve_kJ
    quang_duong_km: float
    traffic_status: string, e.g. 'Tắc đường', 'Đông đúc (chậm)', 'Bình thường', 'Thông thoáng'
    """
    model, scaler_X, scaler_y, label_encoder = load_ann_model()

    # Map correct Vietnamese traffic labels to the numeric encoding used by the saved model
    traffic_map = {
        'Bình thường': 0,
        'Thông thoáng': 1,
        'Tắc đường': 2,
        'Đông đúc (chậm)': 3,
    }
    if traffic_status not in traffic_map:
        raise ValueError(f"Unknown traffic status: {traffic_status}")
    traffic_value = traffic_map[traffic_status]

    x = np.array([
        energy_features['E_roll_kJ'],
        energy_features['E_aero_kJ'],
        energy_features['E_inertia_kJ'],
        energy_features['E_grade_kJ'],
        energy_features['E_curve_kJ'],
        quang_duong_km,
        traffic_value
    ], dtype=float).reshape(1, -1)

    x_scaled = scaler_X.transform(x)
    y_pred_scaled = model.predict(x_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
    return float(max(y_pred, 0.0))


if __name__ == '__main__':
    example = {
        'E_roll_kJ': 1000.0,
        'E_aero_kJ': 500.0,
        'E_inertia_kJ': 50.0,
        'E_grade_kJ': 200.0,
        'E_curve_kJ': 10.0,
    }
    fuel = predict_fuel(example, quang_duong_km=5.0, traffic_status='Bình thường')
    print('Predicted fuel (liters):', fuel)
