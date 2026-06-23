import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mo_hinh_ann_7_features.keras')
SCALER_X_PATH = os.path.join(BASE_DIR, 'scaler_X_7features.pkl')
SCALER_Y_PATH = os.path.join(BASE_DIR, 'scaler_y_7features.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder_trang_thai.pkl')


def assert_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model file not found: {path}")


def load_ann_model():
    for path in (MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, LABEL_ENCODER_PATH):
        assert_file_exists(path)
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
    try:
        model, scaler_X, scaler_y, label_encoder = load_ann_model()
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {e}")
        raise

    # Map correct Vietnamese traffic labels to the numeric encoding used by the saved model
    traffic_map = {
        'Bình thường': 0,
        'Thông thoáng': 1,
        'Tắc đường': 2,
        'Đông đúc (chậm)': 3,
    }
    if traffic_status not in traffic_map:
        print(f"⚠️ Cảnh báo: Trạng thái giao thông '{traffic_status}' không được nhận diện. Sử dụng 'Bình thường' làm mặc định.")
        traffic_value = 0  # Default to 'Bình thường'
    else:
        traffic_value = traffic_map[traffic_status]

    try:
        # Kiểm tra giá trị input
        for key in ['E_roll_kJ', 'E_aero_kJ', 'E_inertia_kJ', 'E_grade_kJ', 'E_curve_kJ']:
            if key not in energy_features or energy_features[key] is None:
                print(f"❌ Lỗi: {key} bị thiếu hoặc None")
                raise ValueError(f"Thiếu hoặc None: {key}")
        
        x = np.array([
            float(energy_features['E_roll_kJ']),
            float(energy_features['E_aero_kJ']),
            float(energy_features['E_inertia_kJ']),
            float(energy_features['E_grade_kJ']),
            float(energy_features['E_curve_kJ']),
            float(quang_duong_km),
            float(traffic_value)
        ], dtype=float).reshape(1, -1)

        # Kiểm tra NaN
        if np.isnan(x).any():
            print(f"❌ Lỗi: Dữ liệu input chứa NaN: {x}")
            raise ValueError("Dữ liệu input chứa NaN")

        x_scaled = scaler_X.transform(x)
        y_pred_scaled = model.predict(x_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
        
        # Kiểm tra kết quả dự đoán
        if np.isnan(y_pred) or y_pred < 0:
            print(f"⚠️ Cảnh báo: Dự đoán bị lỗi {y_pred}. Trả về 0.5 lít mặc định.")
            return 0.5
        
        return float(max(y_pred, 0.0))
    except Exception as e:
        print(f"❌ Lỗi tính toán dự đoán: {e}")
        raise


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
