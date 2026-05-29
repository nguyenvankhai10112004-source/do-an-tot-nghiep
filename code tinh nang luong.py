import numpy as np
import pandas as pd
import gpxpy
import math
import glob
import os
from geopy.distance import distance

# ===================== THÔNG SỐ XE =====================
class ThongSoXe:
    def __init__(self):
        self.m = 1600
        self.f = 0.012
        self.g = 9.81
        self.Cd = 0.34
        self.A = 2.5
        self.xi = 1.05
        self.it = 5.0
        self.rho_std = 1.184
        self.P_std = 101.3
        self.T_std = 293.15

# ===================== HÀM ĐỌC GPX =====================
def read_gpx_to_dataframe(gpx_file_path):
    with open(gpx_file_path, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'time': point.time,
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation if point.elevation else 0
                })
    return pd.DataFrame(points)

# ===================== TÍNH KHOẢNG CÁCH =====================
def tinh_khoang_cach(df):
    coords = list(zip(df['lat'], df['lon']))
    distances = [0]
    for i in range(1, len(coords)):
        d = distance(coords[i-1], coords[i]).meters
        distances.append(d)
    df['s_m'] = distances
    df['s_km'] = df['s_m'].cumsum() / 1000
    return df

# ===================== TÍNH GÓC DỐC =====================
def tinh_goc_doc(df):
    alpha = [0]
    for i in range(1, len(df)):
        dh = df.loc[i, 'ele'] - df.loc[i-1, 'ele']
        ds = df.loc[i, 's_m']
        if ds > 0:
            alpha.append(np.arctan(dh / ds))
        else:
            alpha.append(0)
    df['alpha_rad'] = alpha
    return df

# ===================== TÍNH VẬN TỐC =====================
def tinh_van_toc(df):
    speeds = [0]
    accelerations = [0]
    for i in range(1, len(df)):
        ds = df.loc[i, 's_m']
        dt = (df.loc[i, 'time'] - df.loc[i-1, 'time']).total_seconds()
        if dt > 0:
            v = ds / dt
            a = (v - speeds[-1]) / dt if i > 1 else 0
        else:
            v = 0
            a = 0
        speeds.append(v)
        accelerations.append(a)
    df['v_ms'] = speeds
    df['a_ms2'] = accelerations
    return df

# ===================== PHÂN TÍCH GIAO THÔNG =====================
def classify_traffic(speed_kmh):
    if speed_kmh < 5:
        return "Tắc đường"
    elif speed_kmh < 15:
        return "Đông đúc (chậm)"
    elif speed_kmh < 30:
        return "Bình thường"
    else:
        return "Thông thoáng"

def analyze_traffic_advanced(df):
    df_traffic = df.copy()
    df_traffic['v_kmh'] = df_traffic['v_ms'] * 3.6
    df_traffic['traffic_status'] = df_traffic['v_kmh'].apply(classify_traffic)
    
    total_distance = df_traffic['s_m'].sum()
    
    if total_distance > 0:
        weighted_avg_speed = (df_traffic['v_kmh'] * df_traffic['s_m']).sum() / total_distance
    else:
        weighted_avg_speed = 0
    
    congestion_distance = df_traffic[df_traffic['v_kmh'] < 10]['s_m'].sum()
    slow_distance = df_traffic[(df_traffic['v_kmh'] >= 10) & (df_traffic['v_kmh'] < 25)]['s_m'].sum()
    normal_distance = df_traffic[(df_traffic['v_kmh'] >= 25) & (df_traffic['v_kmh'] < 50)]['s_m'].sum()
    free_distance = df_traffic[df_traffic['v_kmh'] >= 50]['s_m'].sum()
    
    status_distances = {
        'Tắc đường': congestion_distance,
        'Đông đúc (chậm)': slow_distance,
        'Bình thường': normal_distance,
        'Thông thoáng': free_distance
    }
    common_status = max(status_distances, key=status_distances.get) if total_distance > 0 else "Không xác định"
    
    return {
        'avg_speed_kmh': round(weighted_avg_speed, 1),
        'congestion_ratio': round((congestion_distance / total_distance * 100), 1) if total_distance > 0 else 0,
        'slow_ratio': round((slow_distance / total_distance * 100), 1) if total_distance > 0 else 0,
        'normal_ratio': round((normal_distance / total_distance * 100), 1) if total_distance > 0 else 0,
        'free_ratio': round((free_distance / total_distance * 100), 1) if total_distance > 0 else 0,
        'common_status': common_status
    }

# ===================== TÍNH NHIỆT ĐỘ, ÁP SUẤT, MẬT ĐỘ KHÔNG KHÍ THEO ĐỘ CAO =====================
def tinh_nhiet_do_theo_do_cao(elevation_m):
    T0 = 288.15
    L = 0.0065
    T_kelvin = T0 - L * elevation_m
    return max(T_kelvin, 223.15)

def tinh_ap_suat_theo_do_cao(elevation_m):
    P0 = 101325
    L = 0.0065
    T0 = 288.15
    g = 9.80665
    M = 0.0289644
    R = 8.31432
    
    exponent = (g * M) / (R * L)
    factor = 1 - (L * elevation_m / T0)
    
    if factor > 0:
        P = P0 * (factor ** exponent)
    else:
        P = P0 * 0.001
    return P

def tinh_mat_do_khong_khi_tu_do_cao(elevation_m):
    R_air = 287.0
    T = tinh_nhiet_do_theo_do_cao(elevation_m)
    P = tinh_ap_suat_theo_do_cao(elevation_m)
    rho = P / (R_air * T)
    return rho

# ===================== TÍNH BÁN KÍNH ĐƯỜNG CONG =====================
def tinh_ban_kinh_cong(df):
    n = len(df)
    R_values = [float('inf')] * n
    for i in range(1, n-1):
        lat1, lon1 = df.loc[i-1, 'lat'], df.loc[i-1, 'lon']
        lat2, lon2 = df.loc[i, 'lat'], df.loc[i, 'lon']
        lat3, lon3 = df.loc[i+1, 'lat'], df.loc[i+1, 'lon']
        a = distance((lat2, lon2), (lat3, lon3)).meters
        b = distance((lat1, lon1), (lat3, lon3)).meters
        c = distance((lat1, lon1), (lat2, lon2)).meters
        if a + b > c and a + c > b and b + c > a:
            s = (a + b + c) / 2
            area = math.sqrt(s * (s-a) * (s-b) * (s-c))
            if area > 0:
                R = (a * b * c) / (4 * area)
                if 50 <= R <= 5000:
                    R_values[i] = R
    df['R_m'] = R_values
    return df

# ===================== TÍNH CÁC LOẠI NĂNG LƯỢNG =====================
def tinh_nang_luong_chi_tiet(df, ts):
    elevation_avg = df['ele'].mean()
    rho = tinh_mat_do_khong_khi_tu_do_cao(elevation_avg)
    
    df = tinh_ban_kinh_cong(df)
    df['E_roll_kJ'] = 0.0
    df['E_aero_kJ'] = 0.0
    df['E_inertia_kJ'] = 0.0
    df['E_grade_kJ'] = 0.0
    df['E_curve_kJ'] = 0.0
    
    for i in range(1, len(df)):
        s = df.loc[i, 's_m']
        if s <= 0:
            continue
        alpha = df.loc[i, 'alpha_rad']
        v = df.loc[i, 'v_ms']
        a = df.loc[i, 'a_ms2']
        R = df.loc[i, 'R_m']
        
        F_roll = ts.f * ts.m * ts.g * np.cos(alpha)
        df.loc[i, 'E_roll_kJ'] = F_roll * s / 1000
        
        F_aero = 0.5 * rho * ts.Cd * ts.A * (v ** 2)
        df.loc[i, 'E_aero_kJ'] = F_aero * s / 1000
        
        F_inertia = ts.m * a * (ts.xi + 0.0015 * (ts.it ** 2))
        df.loc[i, 'E_inertia_kJ'] = max(0, F_inertia * s / 1000)
        
        if alpha > 0:
            df.loc[i, 'E_grade_kJ'] = ts.m * ts.g * s * np.sin(alpha) / 1000
        
        if R != float('inf') and R < 500 and v > 0:
            F_curve = ts.m * (v ** 2 / R)
            df.loc[i, 'E_curve_kJ'] = F_curve * s / 1000
    
    return df

# ===================== TÍNH NHIÊN LIỆU TỪ NĂNG LƯỢNG =====================
def tinh_nhien_lieu(tong_nang_luong_kJ, hieu_suat=0.28, nang_luong_1_lit_xang=33000):
    return tong_nang_luong_kJ / hieu_suat / nang_luong_1_lit_xang

# ===================== CHƯƠNG TRÌNH CHÍNH =====================
if __name__ == "__main__":
    ts = ThongSoXe()
    
    thu_muc_gpx = "C:/Users/khai/Downloads/đồ án tn/file gpx"
    
    danh_sach_file = glob.glob(os.path.join(thu_muc_gpx, "*.gpx"))
    
    if len(danh_sach_file) == 0:
        print(f"❌ Không tìm thấy file GPX nào trong thư mục: {thu_muc_gpx}")
        exit()
    
    print(f"✅ TÌM THẤY {len(danh_sach_file)} FILE GPX")
    print(f"⚠️ CHỈ GIỮ LẠI CÁC FILE CÓ QUÃNG ĐƯỜNG >= 1 KM\n")
    
    ket_qua_tong_hop = []
    so_file_bi_bo_qua = 0
    
    for idx, gpx_file in enumerate(danh_sach_file):
        ten_file = os.path.basename(gpx_file)
        print(f"[{idx+1}/{len(danh_sach_file)}] Đang xử lý: {ten_file}")
        
        try:
            df = read_gpx_to_dataframe(gpx_file)
            
            df = tinh_khoang_cach(df)
            df = tinh_goc_doc(df)
            df = tinh_van_toc(df)
            df = tinh_nang_luong_chi_tiet(df, ts)
            
            quang_duong_km = df['s_km'].iloc[-1]
            
            # Bỏ qua file có quãng đường < 1 km
            if quang_duong_km < 1.0:
                print(f"   ⏩ Bỏ qua (quãng đường {quang_duong_km:.2f} km < 1 km)\n")
                so_file_bi_bo_qua += 1
                continue
            
            tong_nang_luong = df[['E_roll_kJ', 'E_aero_kJ', 'E_inertia_kJ', 'E_grade_kJ', 'E_curve_kJ']].sum().sum()
            nhien_lieu_lit = tinh_nhien_lieu(tong_nang_luong)
            traffic = analyze_traffic_advanced(df)
            
            ket_qua_tong_hop.append({
                'Ten_file': ten_file,
                'Quang_duong_km': round(quang_duong_km, 2),
                'E_roll_kJ': round(df['E_roll_kJ'].sum(), 2),
                'E_aero_kJ': round(df['E_aero_kJ'].sum(), 2),
                'E_inertia_kJ': round(df['E_inertia_kJ'].sum(), 2),
                'E_grade_kJ': round(df['E_grade_kJ'].sum(), 2),
                'E_curve_kJ': round(df['E_curve_kJ'].sum(), 2),
                'Tong_nang_luong_kJ': round(tong_nang_luong, 2),
                'Nhien_lieu_uoc_luong_lit': round(nhien_lieu_lit, 4),
                'Van_toc_TB_kmh': traffic['avg_speed_kmh'],
                'Ty_le_tac_duong_%': traffic['congestion_ratio'],
                'Ty_le_cham_%': traffic['slow_ratio'],
                'Ty_le_binh_thuong_%': traffic['normal_ratio'],
                'Ty_le_thong_thoang_%': traffic['free_ratio'],
                'Trang_thai_chinh': traffic['common_status']
            })
            
            print(f"   ✅ Hoàn thành: {quang_duong_km:.2f} km, Nhiên liệu: {nhien_lieu_lit:.4f} lít\n")
            
        except Exception as e:
            print(f"   ❌ Lỗi xử lý: {e}\n")
            so_file_bi_bo_qua += 1
            continue
    
    # Tạo DataFrame từ kết quả
    df_tong_hop = pd.DataFrame(ket_qua_tong_hop)
    
    if len(df_tong_hop) == 0:
        print("\n⚠️ KHÔNG CÓ FILE NÀO ĐỦ ĐIỀU KIỆN (>= 1 km)")
    else:
        df_tong_hop = df_tong_hop.sort_values('Ten_file')
        df_tong_hop.to_csv('tong_hop_nang_luong.csv', index=False, encoding='utf-8-sig')
        
        print("="*70)
        print(f"📊 KẾT QUẢ TỔNG HỢP (Giữ lại {len(df_tong_hop)}/{len(danh_sach_file)} file, bỏ qua {so_file_bi_bo_qua} file < 1km)")
        print("="*70)
        print(df_tong_hop.to_string())
        print(f"\n✅ Đã lưu vào: tong_hop_nang_luong.csv")