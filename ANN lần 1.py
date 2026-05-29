import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import joblib

warnings.filterwarnings('ignore')

# ===================== 1. ĐỌC DỮ LIỆU =====================
print("="*70)
print("ANN - DỰ ĐOÁN NHIÊN LIỆU TIÊU THỤ (1 lớp ẩn - 80 nơ-ron)")
print("Đặc trưng: 5 năng lượng + Quãng đường + Trạng thái giao thông")
print("="*70)

# Đọc file CSV với encoding latin1
file_path = r"c:\Users\khai\tong_hop_nang_luong.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Làm sạch tên cột
df.columns = df.columns.str.strip()

print(f"\n📋 Các cột trong file: {df.columns.tolist()}")
print(f"📊 Kích thước dữ liệu: {df.shape[0]} mẫu, {df.shape[1]} cột")

# ===================== 2. SỬA LỖI ENCODING CHO CỘT Trang_thai_chinh =====================
if 'Trang_thai_chinh' in df.columns:
    # Bảng ánh xạ sửa lỗi ký tự
    fix_map = {
        'T?c ???ng': 'Tắc đường',
        '?ông ?úc (ch?m)': 'Đông đúc (chậm)',
        'Bình th??ng': 'Bình thường',
        'Thông thoáng': 'Thông thoáng'
    }
    
    # In giá trị cũ để kiểm tra
    print("\n🔍 Giá trị cũ của cột Trang_thai_chinh:")
    print(f"   {df['Trang_thai_chinh'].unique()}")
    
    # Sửa lỗi
    df['Trang_thai_chinh'] = df['Trang_thai_chinh'].replace(fix_map)
    
    # In giá trị mới để kiểm tra
    print("\n✅ Giá trị sau khi sửa:")
    print(f"   {df['Trang_thai_chinh'].unique()}")
else:
    print("❌ Không tìm thấy cột Trang_thai_chinh!")
    exit()

# ===================== 3. KIỂM TRA CÁC CỘT CẦN THIẾT =====================
# Các cột đặc trưng cần có
feature_columns = ['E_roll_kJ', 'E_aero_kJ', 'E_inertia_kJ', 'E_grade_kJ', 'E_curve_kJ', 'Quang_duong_km', 'Trang_thai_chinh']

# Kiểm tra các cột có tồn tại không
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"❌ Thiếu các cột: {missing_cols}")
    print(f"📋 Các cột hiện có: {df.columns.tolist()}")
    print("\n👉 Hãy đảm bảo file CSV có đủ các cột sau:")
    print("   - E_roll_kJ, E_aero_kJ, E_inertia_kJ, E_grade_kJ, E_curve_kJ")
    print("   - Quang_duong_km, Trang_thai_chinh")
    exit()

# Chọn cột làm nhãn (nhiên liệu)
target_col = None
for col in ['Nhien_lieu_uoc_luong_lit', 'fuel_liters', 'Nhiên liệu', 'Fuel']:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print(f"❌ Không tìm thấy cột nhiên liệu!")
    print(f"📋 Các cột hiện có: {df.columns.tolist()}")
    exit()

# ===================== 4. XỬ LÝ DỮ LIỆU =====================
# Lấy đặc trưng (X)
X_raw = df[feature_columns].copy()

# Xử lý cột Trang_thai_chinh (chuyển từ text sang số)
label_encoder = LabelEncoder()
X_raw['Trang_thai_chinh'] = label_encoder.fit_transform(X_raw['Trang_thai_chinh'])

# In mapping để biết
print("\n📊 MAPPING TRẠNG THÁI GIAO THÔNG:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"   {cls} -> {i}")

X = X_raw.values.astype(float)
y = df[target_col].values.astype(float)

print(f"\n📊 Đặc trưng (X): {feature_columns}")
print(f"   - 5 cột năng lượng (kJ)")
print(f"   - Quãng đường (km)")
print(f"   - Trạng thái giao thông (đã mã hóa số)")
print(f"📊 Nhãn (y): {target_col}")
print(f"📊 Số mẫu ban đầu: {len(X)}")

# ===================== 5. BỘ LỌC: LOẠI BỎ NHIÊN LIỆU ≤ 1 LÍT =====================
print("\n" + "="*60)
print("🔧 TIẾN HÀNH LỌC DỮ LIỆU (NHIÊN LIỆU > 1 LÍT)")
print("="*60)

# Thống kê trước khi lọc
print(f"\n📊 Trước khi lọc:")
print(f"   Số mẫu: {len(X)}")
print(f"   Min nhiên liệu: {y.min():.6f} lít")
print(f"   Max nhiên liệu: {y.max():.6f} lít")
print(f"   Mean nhiên liệu: {y.mean():.6f} lít")

# Lọc bỏ các mẫu có nhiên liệu ≤ 1 lít
mask = y > 1.0
n_removed = np.sum(~mask)
X = X[mask]
y = y[mask]

# Thống kê sau khi lọc
print(f"\n📊 Sau khi lọc (chỉ giữ nhiên liệu > 1 lít):")
print(f"   Số mẫu: {len(X)}")
print(f"   Đã loại bỏ: {n_removed} mẫu")
print(f"   Min nhiên liệu: {y.min():.6f} lít")
print(f"   Max nhiên liệu: {y.max():.6f} lít")
print(f"   Mean nhiên liệu: {y.mean():.6f} lít")

if len(X) == 0:
    print("\n❌ KHÔNG CÒN MẪU NÀO SAU KHI LỌC! Vui lòng giảm ngưỡng lọc.")
    exit()

# ===================== 6. CHIA DỮ LIỆU =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\n📁 Train: {len(X_train)} mẫu")
print(f"📁 Validation: {len(X_val)} mẫu")
print(f"📁 Test: {len(X_test)} mẫu")

# ===================== 7. CHUẨN HÓA =====================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

# ===================== 8. XÂY DỰNG MÔ HÌNH (1 lớp ẩn - 80 nơ-ron) =====================
model = Sequential([
    Dense(80, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Giảm dropout vì chỉ có 1 lớp ẩn
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ===================== 9. HUẤN LUYỆN =====================
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

print("\n🚀 Bắt đầu huấn luyện...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=4,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ===================== 10. ĐÁNH GIÁ =====================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Xử lý giá trị âm (nếu có)
y_pred = np.maximum(y_pred, 0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
print("="*60)
print(f"MAE:  {mae:.6f} lít")
print(f"RMSE: {rmse:.6f} lít")
print(f"R²:   {r2:.6f}")

# ===================== 11. LƯU MÔ HÌNH =====================
model.save('mo_hinh_ann_7_features.keras')
joblib.dump(scaler_X, 'scaler_X_7features.pkl')
joblib.dump(scaler_y, 'scaler_y_7features.pkl')
joblib.dump(label_encoder, 'label_encoder_trang_thai.pkl')

print("\n✅ ĐÃ LƯU MÔ HÌNH THÀNH CÔNG!")
print("   - mo_hinh_ann_7_features.keras")
print("   - scaler_X_7features.pkl")
print("   - scaler_y_7features.pkl")
print("   - label_encoder_trang_thai.pkl")
print("\n✅ HOÀN THÀNH!")
