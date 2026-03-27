"""
ParkInsight startup script.

Default  : Runs Flask on HTTPS (self-signed cert).
           Android Chrome: accept the security warning once → motion sensors work.
--ngrok  : Runs Flask on HTTP + creates ngrok HTTPS tunnel (easiest for iPhone).
"""
import os
import sys
import socket
import threading
import argparse
import webbrowser
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, 'backend')
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND_DIR)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ngrok', action='store_true',
                    help='HTTPS via ngrok tunnel (easiest for iPhone)')
args = parser.parse_args()

local_ip = get_local_ip()

# ── Import Flask app ──────────────────────────────────────────────────────────
# Do this in the main thread so any import errors are immediately visible.
os.chdir(ROOT)
from backend.app import app
import backend.app as _app_module

# ── SSL setup (non-ngrok mode) ────────────────────────────────────────────────
ssl_ctx = None
proto   = 'http'

if not args.ngrok:
    try:
        from gen_cert import generate, CERT_FILE, KEY_FILE
        generate(local_ip)
        import ssl as _ssl
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT_FILE, KEY_FILE)
        _app_module.HTTPS_ENABLED = True
        ssl_ctx = ctx
        proto   = 'https'
        print(f"\nCertificate ready: {CERT_FILE}")
    except Exception as e:
        print(f"\nWARNING: Could not generate SSL cert ({e})")
        print("Falling back to HTTP. Motion sensors may not work on the phone.")
        proto = 'http'

# ── ngrok setup (background thread) ──────────────────────────────────────────
if args.ngrok:
    def _setup_ngrok():
        time.sleep(2.5)  # wait for Flask to be ready
        try:
            from pyngrok import ngrok

            token_paths = [
                os.path.expanduser('~/.config/ngrok/ngrok.yml'),   # Linux/Mac
                os.path.expanduser('~/.ngrok2/ngrok.yml'),          # old ngrok
                os.path.join(os.environ.get('LOCALAPPDATA', ''),
                             'ngrok', 'ngrok.yml'),                  # Windows
                os.path.join(os.environ.get('APPDATA', ''),
                             'ngrok', 'ngrok.yml'),                  # Windows alt
            ]
            if not any(os.path.exists(p) for p in token_paths):
                print("\n  ngrok auth token not found. One-time setup:")
                print("  1. Sign up at https://dashboard.ngrok.com/signup")
                print("  2. Copy your authtoken")
                print("  3. Run:  ngrok config add-authtoken YOUR_TOKEN")
                print("  Then re-run:  python start.py --ngrok\n")
                return

            tunnel    = ngrok.connect(5000)
            https_url = tunnel.public_url
            if https_url.startswith('http://'):
                https_url = 'https://' + https_url[7:]

            os.environ['PARKINSIGHT_BASE_URL'] = https_url
            print(f"\n  ngrok tunnel ready!")
            print(f"  Phone URL : {https_url}/phone  (scan QR on dashboard)\n")

        except Exception as e:
            print(f"\n  ngrok error: {e}\n")

    threading.Thread(target=_setup_ngrok, daemon=True).start()

# ── Open browser (background thread) ─────────────────────────────────────────
def _open_browser():
    time.sleep(2.0)
    # Use localhost for the laptop — Chrome handles self-signed certs on
    # localhost more cleanly than on raw IP addresses.
    host = 'localhost' if proto == 'https' else local_ip
    url = f'{proto}://{host}:5000/dashboard'
    print(f"  Opening: {url}")
    webbrowser.open(url)

threading.Thread(target=_open_browser, daemon=True).start()

# ── Print startup info ────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  ParkInsight")
print("=" * 58)
if args.ngrok:
    print(f"  Dashboard : http://{local_ip}:5000/dashboard")
    print(f"  Phone URL : (ngrok URL printing in a moment...)")
else:
    print(f"  Dashboard : https://{local_ip}:5000/dashboard")
    print(f"  Phone URL : https://{local_ip}:5000/phone")
    print()
    print("  First visit on each device: click 'Advanced' then")
    print("  'Proceed to ... (unsafe)' to accept the self-signed cert.")
print("=" * 58 + "\n")

# ── Run Flask (main thread — errors will be visible here) ────────────────────
try:
    app.run(host='0.0.0.0', port=5000, debug=False,
            use_reloader=False, threaded=True, ssl_context=ssl_ctx)
except Exception as e:
    print(f"\nFlask failed to start: {e}")
    if 'ssl' in str(e).lower() or 'certificate' in str(e).lower():
        print("SSL error — try: pip install pyopenssl")
    elif 'address already in use' in str(e).lower() or '10048' in str(e):
        print("Port 5000 is already in use.")
        print("Kill the other process first, then re-run start.py")
    sys.exit(1)
