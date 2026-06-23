import os
import sys
import json
import importlib.util
from flask import Flask, request, jsonify, send_from_directory
import openrouteservice
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import distance

# Ensure the parent folder is on Python path so predict_fuel_ann can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

WEB_DIR = os.path.join(ROOT_DIR, 'web')

# Import ANN prediction helper
from predict_fuel_ann import predict_fuel

app = Flask(__name__, static_folder=WEB_DIR, static_url_path='/')

# Load energy calculation module from existing file with spaces in name
ENERGY_PATH = os.path.abspath(os.path.join(ROOT_DIR, 'code tinh nang luong.py'))
if not os.path.exists(ENERGY_PATH):
    raise FileNotFoundError(f"Could not find energy file at {ENERGY_PATH}")
spec = importlib.util.spec_from_file_location("energy_src", ENERGY_PATH)
energy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(energy)

VEHICLE_FILE = os.path.join(ROOT_DIR, 'Loại_xe.csv')
MODEL_FILES = {
    'ANN model': os.path.join(ROOT_DIR, 'mo_hinh_ann_7_features.keras'),
    'Scaler X': os.path.join(ROOT_DIR, 'scaler_X_7features.pkl'),
    'Scaler y': os.path.join(ROOT_DIR, 'scaler_y_7features.pkl'),
    'Label encoder': os.path.join(ROOT_DIR, 'label_encoder_trang_thai.pkl'),
}
for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {name} at {path}")
if not os.path.exists(VEHICLE_FILE):
    raise FileNotFoundError(f"Could not find vehicle data file at {VEHICLE_FILE}")

# ORS client
ORS_KEY = os.getenv('ORS_API_KEY')
if not ORS_KEY:
    print("Warning: ORS_API_KEY not set. Set environment variable before running.")
client = openrouteservice.Client(key=ORS_KEY) if ORS_KEY else None


@app.route('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')


@app.route('/vehicles', methods=['GET'])
def get_vehicle_list():
    try:
        vehicle_names = energy.get_vehicle_names(VEHICLE_FILE)
        return jsonify({'vehicles': vehicle_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    start = data.get('start')  # [lat, lon]
    end = data.get('end')
    if not start or not end:
        return jsonify({'error': 'start and end required'}), 400
    if client is None:
        return jsonify({'error': 'ORS_API_KEY not configured on server'}), 500

    coords = [[start[1], start[0]], [end[1], end[0]]]  # ORS expects [lon,lat]
    res = client.directions(coords, profile='driving-car', format='geojson')
    # Extract duration (seconds) from properties if available
    duration = 0
    try:
        segments = res['features'][0]['properties'].get('segments', [])
        for seg in segments:
            duration += seg.get('duration', 0)
    except Exception:
        duration = 0

    return jsonify({'route': res['features'][0], 'duration_s': int(duration)})


@app.route('/compute', methods=['POST'])
def compute_energy():
    payload = request.get_json()
    route_feature = payload.get('route')  # geojson feature or list of coords
    duration_s = payload.get('duration_s', None)
    vehicle_type = payload.get('vehicle_type')

    if route_feature is None:
        return jsonify({'error': 'route required'}), 400
    if not vehicle_type:
        return jsonify({'error': 'vehicle_type required'}), 400

    # Extract coordinates list as (lat, lon)
    geom = route_feature.get('geometry') if isinstance(route_feature, dict) else None
    if geom and geom.get('coordinates'):
        coords = [(c[1], c[0]) for c in geom['coordinates']]
    elif isinstance(route_feature, list):
        coords = [(c[0], c[1]) for c in route_feature]
    else:
        return jsonify({'error': 'invalid route geometry'}), 400

    # Get elevations for each point (may be rate-limited)
    elevs = []
    if client is None:
        return jsonify({'error': 'ORS_API_KEY not configured on server'}), 500
    for lat, lon in coords:
        try:
            e = client.elevation_point((lon, lat))
            # response geometry coordinates: [lon, lat, ele]
            ele = e.get('geometry', {}).get('coordinates', [None, None, 0])[2]
        except Exception:
            ele = 0
        elevs.append(ele)

    # Build timestamps distributed along distance using geopy
    dists = [0.0]
    for i in range(1, len(coords)):
        d = distance(coords[i-1], coords[i]).meters
        dists.append(d)
    total_dist = sum(dists)
    if duration_s is None or duration_s <= 0:
        duration_s = max(int(total_dist / 10), 60)  # heuristic: assume average ~10 m/s

    t0 = datetime.utcnow()
    times = []
    cum = 0.0
    for ds in dists:
        cum += ds
        frac = (cum / total_dist) if total_dist > 0 else 0
        times.append(t0 + timedelta(seconds=int(frac * duration_s)))

    df = pd.DataFrame({'time': times, 'lat': [p[0] for p in coords], 'lon': [p[1] for p in coords], 'ele': elevs})

    # Call energy functions from the loaded module
    df = energy.tinh_khoang_cach(df)
    df = energy.tinh_goc_doc(df)
    df = energy.tinh_van_toc(df)
    try:
        ts = energy.ThongSoXe(loai_xe=vehicle_type, loai_xe_file=VEHICLE_FILE, interactive=False)
    except Exception as e:
        return jsonify({'error': f'Invalid vehicle type: {e}'}), 400
    df = energy.tinh_nang_luong_chi_tiet(df, ts)

    # Traffic and route summary used by ANN
    traffic = energy.analyze_traffic_advanced(df)
    traffic_status = traffic.get('common_status', 'Bình thường')
    quang_duong_km = float(df['s_km'].iloc[-1]) if len(df) > 0 else 0.0
    
    # Kiểm tra quãng đường
    if pd.isna(quang_duong_km) or quang_duong_km is None or quang_duong_km <= 0:
        print(f"⚠️ Cảnh báo: Quãng đường {quang_duong_km} không hợp lệ, set thành 1.0")
        quang_duong_km = 1.0
    
    print(f"📍 Trạng thái giao thông: {traffic_status}")
    print(f"📍 Quãng đường: {quang_duong_km} km")

    # Summarize
    sums = df[['E_roll_kJ', 'E_aero_kJ', 'E_inertia_kJ', 'E_grade_kJ', 'E_curve_kJ']].sum().to_dict()
    total_kJ = sum(sums.values())

    # Kiểm tra NaN values
    print(f"\n📊 Energy sums trước check: {sums}")
    for key, val in sums.items():
        if pd.isna(val) or val is None:
            print(f"⚠️ Cảnh báo: {key} bị NaN, set thành 0")
            sums[key] = 0.0
        else:
            sums[key] = float(sums[key])
    print(f"📊 Energy sums sau check: {sums}")
    total_kJ = sum(sums.values())

    # ANN prediction
    try:
        fuel_ann_l = predict_fuel(
            {
                'E_roll_kJ': float(sums['E_roll_kJ']),
                'E_aero_kJ': float(sums['E_aero_kJ']),
                'E_inertia_kJ': float(sums['E_inertia_kJ']),
                'E_grade_kJ': float(sums['E_grade_kJ']),
                'E_curve_kJ': float(sums['E_curve_kJ']),
            },
            quang_duong_km=float(quang_duong_km),
            traffic_status=traffic_status
        )
        print(f"✅ Dự đoán ANN thành công: {fuel_ann_l} lít")
    except Exception as e:
        print(f"❌ Lỗi ANN prediction: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        fuel_ann_l = None

    resp = {
        'vehicle_type': vehicle_type,
        'sums_kJ': {k: float(round(v, 3)) for k, v in sums.items()},
        'total_kJ': float(round(total_kJ, 3)),
        'fuel_ann_l': float(round(fuel_ann_l, 5)) if fuel_ann_l is not None else None,
        'traffic_status': traffic_status,
        'quang_duong_km': round(quang_duong_km, 3),
        'points': [{'lat': p[0], 'lon': p[1], 'ele': e} for p, e in zip(coords, elevs)]
    }
    return jsonify(resp)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
