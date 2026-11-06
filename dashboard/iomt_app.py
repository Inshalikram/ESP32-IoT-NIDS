# ============================================================
# IoMT 2024 IDS Dashboard — Flask Backend (LOCAL DEPLOYMENT)
#
# Run locally on your PC:
#   pip install flask zeroconf
#   python iomt_app.py
#
# File structure:
#   iomt_app.py
#   templates/
#       iomt_dashboard.html
#
# Auto-detects this PC's WiFi IP and registers mDNS as
# "ids-server.local" so the ESP32 never needs a hardcoded IP —
# even if your router gives the PC a new IP later.
# ============================================================

from flask import Flask, request, jsonify, render_template
from datetime import datetime
from collections import deque, defaultdict
import threading, os, socket, webbrowser, time
from zeroconf import ServiceInfo, Zeroconf

app = Flask(__name__)

# ── In-memory store (resets on restart) ──────────────────────
MAX_EVENTS = 200
lock       = threading.Lock()

events     = deque(maxlen=MAX_EVENTS)   # latest alerts
stats      = {
    "total_flows":    0,
    "total_attacks":  0,
    "benign":         0,
    "DDoS":           0,
    "DoS":            0,
    "Mirai":          0,
    "Recon":          0,
}
attack_timeline = deque(maxlen=60)      # last 60 attack events for chart

# ── Auto-detect this PC's local WiFi IP ───────────────────
def get_wifi_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ── Register mDNS so ESP32 can find PC via ids-server.local ─
# (no need to hardcode/update IP in firmware ever again)
def register_mdns(local_ip, port=5000):
    try:
        ip_bytes = socket.inet_aton(local_ip)
        info = ServiceInfo(
            "_http._tcp.local.",
            "ids-server._http._tcp.local.",
            addresses=[ip_bytes],
            port=port,
            properties={'path': '/iomt_alert'},
            server="ids-server.local."
        )
        zc = Zeroconf()
        zc.register_service(info)
        print(f"mDNS registered: ids-server.local -> {local_ip}:{port}")
        return zc, info
    except Exception as e:
        print(f"mDNS warning (continuing without it): {e}")
        return None, None

# ── Auto-open dashboard in default browser on start ───────
def auto_open_browser(url, delay=1.5):
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not auto-open browser: {e}")
    threading.Thread(target=_open, daemon=True).start()

# ── POST /iomt_alert — ESP32 sends here ──────────────────────
@app.route("/iomt_alert", methods=["POST"])
def iomt_alert():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "no JSON"}), 400

    prediction  = data.get("prediction",  "Unknown")
    confidence  = data.get("confidence",  0.0)
    threat      = data.get("threat",      "NONE")
    src_ip      = data.get("src_ip",      "0.0.0.0")
    dst_ip      = data.get("dst_ip",      "0.0.0.0")
    src_mac     = data.get("src_mac",     "??:??:??:??:??:??")
    dst_mac     = data.get("dst_mac",     "??:??:??:??:??:??")
    protocol    = data.get("protocol",    "?")
    num_packets = data.get("num_packets", 0)
    duration_s  = data.get("duration_s",  0.0)
    syn_count   = data.get("syn_count",   0)
    ack_count   = data.get("ack_count",   0)
    fin_count   = data.get("fin_count",   0)
    rst_count   = data.get("rst_count",   0)
    tot_size    = data.get("tot_size",    0)
    avg_pkt     = data.get("avg_pkt_size",0)

    now = datetime.now().strftime("%H:%M:%S")

    event = {
        "time":        now,
        "prediction":  prediction,
        "confidence":  round(float(confidence), 1),
        "threat":      threat,
        "src_ip":      src_ip,
        "dst_ip":      dst_ip,
        "src_mac":     src_mac,
        "dst_mac":     dst_mac,
        "protocol":    protocol,
        "num_packets": num_packets,
        "duration_s":  round(float(duration_s), 3),
        "syn_count":   syn_count,
        "ack_count":   ack_count,
        "fin_count":   fin_count,
        "rst_count":   rst_count,
        "tot_size":    tot_size,
        "avg_pkt":     round(float(avg_pkt), 1),
    }

    with lock:
        events.appendleft(event)
        stats["total_flows"] += 1

        if prediction == "BenignTraffic":
            stats["benign"] += 1
        else:
            stats["total_attacks"] += 1
            if prediction in stats:
                stats[prediction] += 1
            attack_timeline.appendleft({
                "time":       now,
                "prediction": prediction,
                "confidence": round(float(confidence), 1),
                "src_ip":     src_ip,
                "threat":     threat,
            })

    return jsonify({"status": "ok"}), 200


# ── GET /iomt_data — dashboard polls this ────────────────────
@app.route("/iomt_data", methods=["GET"])
def iomt_data():
    with lock:
        return jsonify({
            "stats":    dict(stats),
            "events":   list(events)[:50],
            "timeline": list(attack_timeline)[:20],
            "updated":  datetime.now().strftime("%H:%M:%S"),
        })


# ── GET /iomt — serve dashboard ───────────────────────────────
@app.route("/iomt", methods=["GET"])
def dashboard():
    return render_template("iomt_dashboard.html")


# ── GET / — also serve dashboard at root for convenience ─────
@app.route("/", methods=["GET"])
def index():
    return render_template("iomt_dashboard.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    local_ip = get_wifi_ip()
    zc, info = register_mdns(local_ip, port)

    print("=" * 50)
    print("  IoMT 2024 IDS Dashboard")
    print("=" * 50)
    print(f"  PC IP        : {local_ip}  (auto-detected, may change)")
    print(f"  mDNS name    : ids-server.local  (use this in firmware — won't change)")
    print(f"  ESP32 config : SERVER_URL = \"http://ids-server.local:{port}/iomt_alert\"")
    print(f"  Dashboard    : http://localhost:{port}")
    print("=" * 50)

    auto_open_browser(f"http://localhost:{port}")

    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    finally:
        if zc and info:
            zc.unregister_service(info)
            zc.close()
