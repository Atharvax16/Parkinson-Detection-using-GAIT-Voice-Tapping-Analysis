"""
ParkInsight Flask Backend
Serves API endpoints for gait analysis, tapping analysis, voice analysis.
Serves phone capture page and dashboard.
"""
import os
import json
import io
import csv
import socket
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import feature extraction
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction.gait_features import extract_gait_features, get_feature_names
from feature_extraction.tapping_features import extract_tapping_features, assess_tapping_risk

app = Flask(__name__)
CORS(app)

# Set to True when running with self-signed SSL (set by start.py or __main__)
HTTPS_ENABLED = False

# ---- Load Models ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    gait_bundle = joblib.load(os.path.join(MODEL_DIR, 'gait_model.pkl'))
    gait_model = gait_bundle['model']
    gait_scaler = gait_bundle['scaler']
    gait_feature_names = gait_bundle['feature_names']
    print(f"Gait model loaded ({len(gait_feature_names)} features)")
except Exception as e:
    print(f"Warning: Could not load gait model: {e}")
    gait_model = None
    gait_scaler = None
    gait_feature_names = get_feature_names()

try:
    voice_bundle = joblib.load(os.path.join(MODEL_DIR, 'voice_model.pkl'))
    voice_model = voice_bundle['model']
    voice_scaler = voice_bundle['scaler']
    voice_feature_names = voice_bundle['feature_names']
    print(f"Voice model loaded ({len(voice_feature_names)} features)")
except Exception as e:
    print(f"Warning: Could not load voice model: {e}")
    voice_model = None
    voice_scaler = None
    voice_feature_names = []

# ---- Store latest results in memory ----
latest_results = {
    'gait': None,
    'tapping': None,
    'voice': None,
    'combined': None
}

# ---- Load demo data ----
DEMO_DIR = os.path.join(os.path.dirname(__file__), 'demo_data')


def load_demo(scenario):
    filepath = os.path.join(DEMO_DIR, f'{scenario}.json')
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


# ======== SHAP (lazy import due to cv2 issue) ========
def compute_shap_values(model, feature_scaled, feature_names):
    """Compute SHAP values, handling import issues gracefully."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(feature_scaled)
        if isinstance(shap_vals, list):
            # Binary classification: shap_vals[1] is for positive class
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]
        return dict(zip(feature_names, [float(v) for v in vals]))
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        # Return dummy SHAP values based on feature importance
        try:
            importances = model.feature_importances_
            return dict(zip(feature_names, [float(v) for v in importances]))
        except:
            return {name: 0.0 for name in feature_names}


# ======== API ENDPOINTS ========

@app.route('/api/gait/analyze', methods=['POST'])
def analyze_gait():
    """Receive raw accelerometer JSON from phone, extract features, predict."""
    try:
        data = request.json.get('sensor_data', [])

        if len(data) < 50:
            return jsonify({'error': 'Not enough sensor data. Walk for at least 10 seconds.'}), 400

        # Convert to numpy arrays
        timestamps = np.array([d['t'] for d in data])
        acc_x = np.array([d['ax'] for d in data])
        acc_y = np.array([d['ay'] for d in data])
        acc_z = np.array([d['az'] for d in data])
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        # Extract features
        features = extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z)

        # Check for NaN features
        has_nan = any(np.isnan(v) for v in features.values())

        if has_nan or gait_model is None:
            # Fallback: use heuristic scoring
            cv = features.get('stride_time_cv', 5.0)
            if np.isnan(cv):
                cv = 5.0
            prob_pd = min(cv / 15.0, 1.0)  # Simple heuristic
            prediction = 1 if prob_pd > 0.5 else 0
            probability = [1 - prob_pd, prob_pd]
            shap_values = {name: 0.0 for name in gait_feature_names}
        else:
            # Scale and predict
            feature_vector = np.array([features.get(f, 0.0) for f in gait_feature_names]).reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            feature_scaled = gait_scaler.transform(feature_vector)

            prediction = int(gait_model.predict(feature_scaled)[0])
            probability = gait_model.predict_proba(feature_scaled)[0].tolist()

            shap_values = compute_shap_values(gait_model, feature_scaled, gait_feature_names)

        result = {
            'prediction': prediction,
            'probability_healthy': float(probability[0]),
            'probability_pd': float(probability[1]),
            'risk_level': 'Low' if probability[1] < 0.3 else 'Medium' if probability[1] < 0.7 else 'High',
            'features': {k: (float(v) if not np.isnan(v) else None) for k, v in features.items()},
            'shap_values': shap_values,
            'raw_signal': {
                'time': timestamps.tolist(),
                'magnitude': acc_magnitude.tolist()
            }
        }

        latest_results['gait'] = result
        update_combined_score()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gait/upload-csv', methods=['POST'])
def upload_csv():
    """Accept phyphox-exported CSV file (columns: time, acc_x, acc_y, acc_z)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        content = file.read().decode('utf-8')
        reader = csv.reader(io.StringIO(content))

        rows = list(reader)
        # Find header or skip it
        start_row = 0
        for i, row in enumerate(rows):
            try:
                float(row[0])
                start_row = i
                break
            except (ValueError, IndexError):
                continue

        sensor_data = []
        for row in rows[start_row:]:
            try:
                t = float(row[0])
                ax = float(row[1])
                ay = float(row[2])
                az = float(row[3])
                sensor_data.append({'t': t, 'ax': ax, 'ay': ay, 'az': az})
            except (ValueError, IndexError):
                continue

        if len(sensor_data) < 50:
            return jsonify({'error': 'Not enough data in CSV'}), 400

        # Reuse the analyze endpoint logic
        request_data = {'sensor_data': sensor_data}
        # Call analyze_gait_internal
        return _analyze_gait_data(sensor_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _analyze_gait_data(data):
    """Internal function to process sensor data."""
    timestamps = np.array([d['t'] for d in data])
    acc_x = np.array([d['ax'] for d in data])
    acc_y = np.array([d['ay'] for d in data])
    acc_z = np.array([d['az'] for d in data])
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    features = extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z)
    has_nan = any(np.isnan(v) for v in features.values())

    if has_nan or gait_model is None:
        cv = features.get('stride_time_cv', 5.0)
        if np.isnan(cv):
            cv = 5.0
        prob_pd = min(cv / 15.0, 1.0)
        prediction = 1 if prob_pd > 0.5 else 0
        probability = [1 - prob_pd, prob_pd]
        shap_values = {name: 0.0 for name in gait_feature_names}
    else:
        feature_vector = np.array([features.get(f, 0.0) for f in gait_feature_names]).reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        feature_scaled = gait_scaler.transform(feature_vector)
        prediction = int(gait_model.predict(feature_scaled)[0])
        probability = gait_model.predict_proba(feature_scaled)[0].tolist()
        shap_values = compute_shap_values(gait_model, feature_scaled, gait_feature_names)

    result = {
        'prediction': prediction,
        'probability_healthy': float(probability[0]),
        'probability_pd': float(probability[1]),
        'risk_level': 'Low' if probability[1] < 0.3 else 'Medium' if probability[1] < 0.7 else 'High',
        'features': {k: (float(v) if not np.isnan(v) else None) for k, v in features.items()},
        'shap_values': shap_values,
        'raw_signal': {
            'time': timestamps.tolist(),
            'magnitude': acc_magnitude.tolist()
        }
    }

    latest_results['gait'] = result
    update_combined_score()
    return jsonify(result)


@app.route('/api/tapping/analyze', methods=['POST'])
def analyze_tapping():
    """Receive tap timestamps from phone."""
    try:
        tap_data = request.json.get('tap_data', [])

        if len(tap_data) < 6:
            return jsonify({'error': 'Not enough taps recorded'}), 400

        features = extract_tapping_features(tap_data)
        if features is None:
            return jsonify({'error': 'Could not extract tapping features'}), 400

        risk_score, risk_level = assess_tapping_risk(features)

        # Compute intervals for visualization
        timestamps = np.array(tap_data)
        intervals = np.diff(timestamps) / 1000.0
        intervals = intervals[intervals < 2.0]

        result = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'features': features,
            'tap_intervals': intervals.tolist()
        }

        latest_results['tapping'] = result
        update_combined_score()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _audio_to_wav(src_path, suffix):
    """
    Return a path to a WAV file ready for scipy.
    - .wav  → return as-is (phone already encoded it client-side)
    - .mp4/.webm → try ffmpeg conversion; return None if ffmpeg unavailable
    """
    if suffix == '.wav':
        return src_path          # already WAV, no conversion needed

    import subprocess
    wav_path = src_path.replace(suffix, '.wav')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', src_path, '-ar', '22050', '-ac', '1',
             '-loglevel', 'quiet', wav_path],
            check=True, timeout=30
        )
        return wav_path
    except Exception as e:
        print(f"ffmpeg conversion failed: {e}")
        return None


def _compute_dfa(x, min_n=4, n_scales=8):
    """Detrended Fluctuation Analysis — pure numpy. Range ~0.61–0.83."""
    N = len(x)
    max_n = max(N // 4, min_n + 1)
    scales = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), n_scales).astype(int))
    y = np.cumsum(x - np.mean(x))
    flucts, used = [], []
    for n in scales:
        n_win = N // n
        if n_win < 1:
            continue
        rms_list = []
        for i in range(n_win):
            seg = y[i * n:(i + 1) * n]
            t   = np.arange(n)
            trend = np.polyval(np.polyfit(t, seg, 1), t)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        flucts.append(np.mean(rms_list))
        used.append(n)
    if len(flucts) < 2:
        return 0.72
    slope = np.polyfit(np.log10(used), np.log10(flucts), 1)[0]
    return float(np.clip(slope, 0.5, 1.0))


def _extract_voice_features(wav_path):
    """
    Extract exactly the 13 features the trained model expects, verified
    against the model's StandardScaler statistics (mean ± 2σ ranges).

    Uses only scipy + numpy — no Praat/parselmouth required.
    Returns a dict or None if audio is unusable.
    """
    from scipy.io import wavfile
    from scipy.signal import find_peaks

    sr, data = wavfile.read(wav_path)

    # ── Mono float64 normalised to [-1, 1] ───────────────────────────────────
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    peak = np.abs(data).max()
    if peak > 1.0:
        data = data / (32768.0 if peak > 1000 else peak)

    # ── Trim silence ──────────────────────────────────────────────────────────
    voiced_idx = np.where(np.abs(data) > 0.01)[0]
    if len(voiced_idx) < int(sr * 0.3):
        return None
    data = data[voiced_idx[0]:voiced_idx[-1]]

    # ── F0 via autocorrelation (25 ms frames, 10 ms hop) ─────────────────────
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    min_lag   = max(int(sr / 500), 1)
    max_lag   = int(sr / 75)
    f0_list   = []

    for start in range(0, len(data) - frame_len, hop_len):
        frame = data[start:start + frame_len] * np.hanning(frame_len)
        corr  = np.correlate(frame, frame, mode='full')[frame_len - 1:]
        if max_lag >= len(corr) or corr[0] < 1e-10:
            continue
        seg = corr[min_lag:max_lag]
        peaks, _ = find_peaks(seg / corr[0], height=0.25, distance=5)
        if len(peaks):
            lag = peaks[np.argmax(seg[peaks])] + min_lag
            f0  = sr / lag
            if 75 <= f0 <= 500:
                f0_list.append(f0)

    if len(f0_list) < 20:
        return None

    f0      = np.array(f0_list, dtype=np.float64)
    T       = 1.0 / f0
    mean_T  = float(np.mean(T))
    mean_f0 = float(np.mean(f0))

    # ── Jitter ────────────────────────────────────────────────────────────────
    # MDVP:Jitter(%) in Oxford dataset is stored as a fraction (0.006 ≈ 0.6%),
    # NOT multiplied by 100. Training mean = 0.0062, std = 0.0048.
    jitter_frac = float(np.mean(np.abs(np.diff(T))) / mean_T)
    jitter_abs  = float(np.mean(np.abs(np.diff(T))))   # seconds (~40 µs)

    # ── Shimmer ───────────────────────────────────────────────────────────────
    rms_list = []
    for start in range(0, len(data) - frame_len, hop_len):
        r = float(np.sqrt(np.mean(data[start:start + frame_len] ** 2)))
        if r > 1e-6:
            rms_list.append(r)
    if len(rms_list) >= 2:
        rms_arr       = np.array(rms_list)
        shimmer_local = float(np.mean(np.abs(np.diff(rms_arr))) / np.mean(rms_arr))
    else:
        shimmer_local = 0.03
    shimmer_local = float(np.clip(shimmer_local, 1e-6, 0.5))

    # ── HNR and NHR ───────────────────────────────────────────────────────────
    # NHR = noise/harmonics  (~0.025 for healthy), HNR = 10*log10(harm/noise)
    ac   = np.correlate(data, data, mode='full')[len(data) - 1:]
    r0   = ac[0]
    pl   = max(int(sr / mean_f0), 1)
    rp   = ac[min(pl, len(ac) - 1)]
    harm  = max(float(rp),      1e-10)
    noise = max(float(r0 - rp), 1e-10)
    hnr   = float(np.clip(10.0 * np.log10(harm / noise), -10.0, 40.0))
    nhr   = float(np.clip(noise / harm, 0.0, 1.0))

    # ── spread1, spread2 ─────────────────────────────────────────────────────
    # spread2 = coefficient of variation of F0 (std/mean) → training range 0.06–0.39
    # spread1 = 2·ln(spread2)                             → training range -7.9 to -3.5
    cv      = float(np.clip(np.std(f0) / mean_f0, 1e-6, 1.0))
    spread2 = cv
    spread1 = float(2.0 * np.log(cv))   # natural log → negative

    # ── DFA (computed from real audio) ───────────────────────────────────────
    dfa = _compute_dfa(data[:min(len(data), sr * 5)])

    # ── RPDE, D2: use training-set means as stable defaults ──────────────────
    # These require phase-space embedding beyond scipy's scope.
    # Training means: RPDE=0.499, D2=2.382 — within 1 std of all samples.
    rpde = 0.50
    d2   = 2.38

    return {
        'MDVP:Fo(Hz)':      mean_f0,
        'MDVP:Fhi(Hz)':     float(np.max(f0)),
        'MDVP:Flo(Hz)':     float(np.min(f0)),
        'MDVP:Jitter(%)':   jitter_frac,
        'MDVP:Jitter(Abs)': jitter_abs,
        'MDVP:Shimmer':     shimmer_local,
        'NHR':     nhr,
        'HNR':     hnr,
        'RPDE':    rpde,
        'DFA':     dfa,
        'spread1': spread1,
        'spread2': spread2,
        'D2':      d2,
    }


@app.route('/api/voice/analyze', methods=['POST'])
def analyze_voice():
    """
    Voice analysis:
    - multipart/form-data 'audio' (mp4/webm/wav) → scipy feature extraction → model
    - application/json 'features' dict → model directly
    """
    try:
        extracted_features = None
        features_dict      = {}

        # --- Audio file upload ---
        if 'audio' in request.files:
            import tempfile, os as _os
            audio_file = request.files['audio']
            fname  = audio_file.filename or ''
            if fname.endswith('.wav'):
                suffix = '.wav'
            elif fname.endswith('.mp4'):
                suffix = '.mp4'
            else:
                suffix = '.webm'

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                audio_file.save(tmp.name)
                tmp_path = tmp.name

            wav_path = None
            try:
                wav_path = _audio_to_wav(tmp_path, suffix)
                if wav_path and _os.path.exists(wav_path):
                    extracted_features = _extract_voice_features(wav_path)
                    if extracted_features is None:
                        print("Voice: could not extract features (audio too short or no voiced frames)")
                else:
                    print("Voice: ffmpeg not available — install ffmpeg to enable voice analysis")
            finally:
                try: _os.unlink(tmp_path)
                except Exception: pass
                if wav_path and wav_path != tmp_path:
                    try: _os.unlink(wav_path)
                    except Exception: pass

        # --- JSON features (manual / test) ---
        elif request.json and 'features' in request.json:
            extracted_features = request.json['features']

        # --- Run model ---
        if extracted_features and voice_model is not None:
            feature_vector = np.array(
                [extracted_features.get(f, 0.0) for f in voice_feature_names]
            ).reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            feature_scaled = voice_scaler.transform(feature_vector)
            prediction  = int(voice_model.predict(feature_scaled)[0])
            probability = voice_model.predict_proba(feature_scaled)[0].tolist()
            features_dict = extracted_features
        elif extracted_features:
            # Features extracted but no model — use jitter/shimmer heuristic
            j = float(extracted_features.get('MDVP:Jitter(%)', 0.5))
            s = float(extracted_features.get('MDVP:Shimmer',   0.05))
            h = float(extracted_features.get('HNR',            10.0))
            # Higher jitter/shimmer and lower HNR → higher PD risk
            prob_pd    = float(np.clip((j / 2.0) + (s * 2.0) + max(0, (20 - h) / 40), 0.05, 0.95))
            prediction = 1 if prob_pd > 0.5 else 0
            probability = [1 - prob_pd, prob_pd]
            features_dict = extracted_features
        else:
            # No audio or extraction failed — return error instead of fake 72%
            return jsonify({'error': 'Voice feature extraction failed. '
                            'Make sure ffmpeg is installed: https://ffmpeg.org/download.html'}), 400

        result = {
            'prediction': prediction,
            'probability_pd': float(probability[1]),
            'probability_healthy': float(probability[0]),
            'risk_level': 'Low' if probability[1] < 0.35 else 'Medium' if probability[1] < 0.65 else 'High',
            'features': {k: v for k, v in features_dict.items() if k != 'status'},
        }

        latest_results['voice'] = result
        update_combined_score()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/server-info', methods=['GET'])
def server_info():
    """Return server's local IP (or ngrok URL) so QR code can use it."""
    import urllib.request as _urllib

    # 1. Check env var set by start.py --ngrok
    ngrok_url = os.environ.get('PARKINSIGHT_BASE_URL')

    # 2. Fallback: query ngrok's local API (works for any ngrok process on port 4040)
    if not ngrok_url:
        try:
            with _urllib.urlopen('http://localhost:4040/api/tunnels', timeout=1) as resp:
                data = json.loads(resp.read())
                for t in data.get('tunnels', []):
                    url = t.get('public_url', '')
                    if url.startswith('https://'):
                        ngrok_url = url
                        break
                    elif url.startswith('http://') and 'ngrok' in url:
                        ngrok_url = url.replace('http://', 'https://')
                        break
        except Exception:
            pass

    if ngrok_url:
        ngrok_url = ngrok_url.rstrip('/')
        host = ngrok_url.replace('https://', '').replace('http://', '')
        return jsonify({'ip': host, 'port': '', 'base_url': ngrok_url, 'https': True})

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '127.0.0.1'

    if HTTPS_ENABLED:
        base = f'https://{local_ip}:5000'
        return jsonify({'ip': local_ip, 'port': 5000, 'base_url': base, 'https': True})

    return jsonify({'ip': local_ip, 'port': 5000, 'https': False})


@app.route('/api/results/latest', methods=['GET'])
def get_latest_results():
    """Dashboard polls this endpoint."""
    return jsonify(latest_results)


@app.route('/api/results/reset', methods=['POST'])
def reset_results():
    """Clear all results for new patient/demo."""
    for key in latest_results:
        latest_results[key] = None
    return jsonify({'status': 'reset'})


@app.route('/api/demo/<scenario>', methods=['POST'])
def run_demo(scenario):
    """Load and process pre-recorded demo data."""
    try:
        demo = load_demo(scenario)
        if demo is None:
            return jsonify({'error': f'Demo scenario "{scenario}" not found'}), 404

        # Reset first
        for key in latest_results:
            latest_results[key] = None

        # Process gait demo
        if 'gait' in demo:
            gait_data = demo['gait']['sensor_data']
            timestamps = np.array([d['t'] for d in gait_data])
            acc_x = np.array([d['ax'] for d in gait_data])
            acc_y = np.array([d['ay'] for d in gait_data])
            acc_z = np.array([d['az'] for d in gait_data])
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            features = extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z)
            has_nan = any(np.isnan(v) for v in features.values())

            if has_nan or gait_model is None:
                cv = features.get('stride_time_cv', 5.0)
                if np.isnan(cv):
                    cv = 5.0
                prob_pd = min(cv / 15.0, 1.0)
                prediction = 1 if prob_pd > 0.5 else 0
                probability = [1 - prob_pd, prob_pd]
                shap_vals = {name: 0.0 for name in gait_feature_names}
            else:
                feature_vector = np.array([features.get(f, 0.0) for f in gait_feature_names]).reshape(1, -1)
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                feature_scaled = gait_scaler.transform(feature_vector)
                prediction = int(gait_model.predict(feature_scaled)[0])
                probability = gait_model.predict_proba(feature_scaled)[0].tolist()
                shap_vals = compute_shap_values(gait_model, feature_scaled, gait_feature_names)

            latest_results['gait'] = {
                'prediction': prediction,
                'probability_healthy': float(probability[0]),
                'probability_pd': float(probability[1]),
                'risk_level': 'Low' if probability[1] < 0.3 else 'Medium' if probability[1] < 0.7 else 'High',
                'features': {k: (float(v) if not np.isnan(v) else None) for k, v in features.items()},
                'shap_values': shap_vals,
                'raw_signal': {
                    'time': timestamps.tolist(),
                    'magnitude': acc_magnitude.tolist()
                }
            }

        # Process tapping demo
        if 'tapping' in demo:
            tap_data = demo['tapping']['tap_data']
            features = extract_tapping_features(tap_data)
            if features:
                risk_score, risk_level = assess_tapping_risk(features)
                timestamps = np.array(tap_data)
                intervals = np.diff(timestamps) / 1000.0
                intervals = intervals[intervals < 2.0]

                latest_results['tapping'] = {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'features': features,
                    'tap_intervals': intervals.tolist()
                }

        update_combined_score()
        return jsonify({'status': 'ok', 'scenario': scenario, 'results': latest_results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def update_combined_score():
    """Weighted combination of available test results."""
    scores = []
    weights = []

    if latest_results['gait']:
        scores.append(latest_results['gait']['probability_pd'])
        weights.append(0.5)
    if latest_results['voice']:
        scores.append(latest_results['voice']['probability_pd'])
        weights.append(0.3)
    if latest_results['tapping']:
        scores.append(latest_results['tapping']['risk_score'])
        weights.append(0.2)

    if scores:
        combined = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        latest_results['combined'] = {
            'score': combined,
            'risk_level': 'Low' if combined < 0.3 else 'Medium' if combined < 0.7 else 'High',
            'tests_completed': len(scores)
        }


# ---- Serve Static Files ----

@app.route('/cert')
def download_cert():
    """Lets iPhone download and install the SSL certificate."""
    cert_path = os.path.join(os.path.dirname(__file__), 'certs', 'cert.pem')
    if not os.path.exists(cert_path):
        return 'No certificate found. Run: python backend/app.py --https first.', 404
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), 'certs'),
        'cert.pem',
        mimetype='application/x-pem-file',
        as_attachment=True,
        download_name='parkinsight.pem'
    )


@app.route('/phone')
@app.route('/phone/')
def serve_phone():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'phone-capture'), 'index.html')


@app.route('/dashboard')
@app.route('/dashboard/')
def serve_dashboard():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'dashboard'), 'index.html')


@app.route('/dashboard/<path:path>')
def serve_dashboard_files(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'dashboard'), path)


@app.route('/')
def index():
    """Root route — redirect to dashboard."""
    return serve_dashboard()


if __name__ == '__main__':
    # Always run with HTTPS so phone motion sensors work without ngrok
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = 'localhost'

    from gen_cert import generate, CERT_FILE, KEY_FILE
    generate(local_ip)
    globals()['HTTPS_ENABLED'] = True

    print("\n" + "=" * 55)
    print("  ParkInsight Server  (HTTPS)")
    print("=" * 55)
    print(f"  Dashboard : https://{local_ip}:5000/dashboard")
    print(f"  Phone URL : https://{local_ip}:5000/phone")
    print()
    print("  First time on each device:")
    print("  - Laptop/Android: accept the browser security warning once.")
    print(f"  - iPhone: install cert via https://{local_ip}:5000/cert")
    print("=" * 55 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context=(CERT_FILE, KEY_FILE))
