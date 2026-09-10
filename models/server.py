# ============================================================
# IoMT IDS — PC Server + MQTT
# ============================================================

from flask import Flask, request, jsonify, render_template_string
import pickle, numpy as np, threading, time, socket, struct
from collections import deque, defaultdict
import paho.mqtt.client as mqtt_client
import json

app = Flask(__name__)

# ── MQTT Setup ────────────────────────────────────────────────
MQTT_BROKER = "1e3008c1b98046c799d9d403f2d212d5.s1.eu.hivemq.cloud"
MQTT_PORT   = 8883
MQTT_USER   = "NIDSystem"
MQTT_PASS   = "Idsproject1"
MQTT_TOPIC  = "ids/alerts"

def on_connect(client, userdata, flags, rc):
    if rc == 0: print("[MQTT] Connected to HiveMQ!")
    else:       print(f"[MQTT] Failed rc={rc}")

def on_publish(client, userdata, mid):
    print(f"[MQTT] Alert published!")

m = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1)
m.username_pw_set(MQTT_USER, MQTT_PASS)
m.tls_set()
m.on_connect = on_connect
m.on_publish  = on_publish
m.connect(MQTT_BROKER, MQTT_PORT)
m.loop_start()

# ── Discovery constants ───────────────────────────────────────
DISCOVERY_PORT     = 5001
DISCOVERY_MAGIC    = b"IOMT_IDS"
DISCOVERY_INTERVAL = 2.0

# ── Load model ────────────────────────────────────────────────
print("Loading model...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_FEATURES = [
    'AVG', 'Covariance', 'Duration', 'Header_Length', 'IAT', 'ICMP',
    'Magnitue', 'Max', 'Min', 'Number', 'Protocol_Type', 'Radius',
    'Std', 'TCP', 'Tot_size', 'UDP', 'Variance', 'Weight',
    'ack_count', 'ack_flag_number', 'fin_count', 'psh_flag_number',
    'rst_count', 'rst_flag_number', 'syn_count', 'syn_flag_number'
]

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded")
except:
    scaler = None
    print("No scaler found — running unscaled")

CLASS_NAMES  = {0:"BenignTraffic", 1:"DDoS", 2:"DoS", 3:"Mirai", 4:"Recon"}
THREAT_LEVEL = {0:"NONE", 1:"CRITICAL", 2:"HIGH", 3:"CRITICAL", 4:"MEDIUM"}

# ── mDNS ──────────────────────────────────────────────────────
def start_mdns(local_ip):
    try:
        from zeroconf import ServiceInfo, Zeroconf
        zc   = Zeroconf()
        info = ServiceInfo(
            "_http._tcp.local.",
            "iomt-ids._http._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=5000,
            properties={"path": "/predict"},
            server="iomt-ids.local.",
        )
        zc.register_service(info)
        print(f"[mDNS] Registered: iomt-ids.local -> {local_ip}")
    except Exception as e:
        print(f"[mDNS] Not available ({e})")

# ── UDP beacon ────────────────────────────────────────────────
def udp_beacon(local_ip):
    msg  = f"IOMT_IDS {local_ip} 5000".encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"[UDP Beacon] Broadcasting every 2s on port {DISCOVERY_PORT}")
    while True:
        try:
            sock.sendto(msg, ("255.255.255.255", DISCOVERY_PORT))
        except Exception as e:
            print(f"[UDP Beacon] {e}")
        time.sleep(2)

print(f"Model ready | Features: {len(MODEL_FEATURES)} | Classes: {model.classes_}")

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return "127.0.0.1"
    finally:
        s.close()

# ── Flow table ────────────────────────────────────────────────
lock = threading.Lock()

def new_flow():
    return {
        "pkts": [], "first_ts": None, "last_ts": None, "prev_ts": None,
        "iat_sum": 0.0, "iat_count": 0,
        "tcp": 0, "udp": 0, "icmp": 0, "hdr_len": 0,
        "syn": 0, "fin": 0, "rst": 0, "ack": 0, "psh": 0, "cwr": 0, "ece": 0,
        "src_mac": "??", "dst_mac": "??"
    }

flow_table = defaultdict(new_flow)
events     = deque(maxlen=300)
timeline   = deque(maxlen=60)
stats = {
    "total_flows": 0, "total_attacks": 0, "benign": 0,
    "DDoS": 0, "DoS": 0, "Mirai": 0, "Recon": 0,
    "packets_received": 0, "packets_processed": 0
}

# ── Packet parser ─────────────────────────────────────────────
def parse_hex_payload(hex_str, pkt_type, src_mac, dst_mac):
    try:
        data = bytes.fromhex(hex_str)
    except Exception:
        return None

    if pkt_type != "DATA" or len(data) < 24:
        return None

    fc0 = data[0]; fc1 = data[1]
    frame_type    = (fc0 >> 2) & 0x03
    frame_subtype = (fc0 >> 4) & 0x0F
    if frame_type != 2:
        return None

    toDS   = bool(fc1 & 0x01)
    fromDS = bool(fc1 & 0x02)
    dot11_hdr = 24
    if frame_subtype & 0x08: dot11_hdr += 2
    if toDS and fromDS:      dot11_hdr += 6

    if len(data) < dot11_hdr + 8:
        return None

    llc = data[dot11_hdr:]
    if len(llc) < 8 or llc[0] != 0xAA or llc[1] != 0xAA or llc[2] != 0x03:
        return None
    ethertype = (llc[6] << 8) | llc[7]
    if ethertype != 0x0800:
        return None

    ip_off = dot11_hdr + 8
    if len(data) < ip_off + 20:
        return None

    ip = data[ip_off:]
    if (ip[0] >> 4) != 4:
        return None

    ip_hdr_len = (ip[0] & 0x0F) * 4
    protocol   = ip[9]
    src_ip     = ".".join(str(b) for b in ip[12:16])
    dst_ip     = ".".join(str(b) for b in ip[16:20])
    pkt_len    = (ip[2] << 8) | ip[3]

    tcp_flags = 0
    if protocol == 6:
        tcp_off = ip_off + ip_hdr_len
        if len(data) >= tcp_off + 14:
            tcp_flags = data[tcp_off + 13]

    if protocol not in (6, 17, 1):
        return None

    return {
        "src_ip": src_ip, "dst_ip": dst_ip,
        "protocol": protocol, "ip_hdr_len": ip_hdr_len,
        "pkt_len": pkt_len, "tcp_flags": tcp_flags,
        "src_mac": src_mac, "dst_mac": dst_mac,
    }

def update_flow(pkt, now):
    key = (pkt["src_ip"], pkt["dst_ip"], pkt["protocol"])
    f   = flow_table[key]
    if f["first_ts"] is None: f["first_ts"] = now
    if f["prev_ts"] is not None and f["prev_ts"] != now:
        f["iat_sum"]   += now - f["prev_ts"]
        f["iat_count"] += 1
    f["prev_ts"] = now
    f["last_ts"] = now
    f["pkts"].append(pkt["pkt_len"])
    f["hdr_len"] += pkt["ip_hdr_len"]
    p = pkt["protocol"]
    if p == 6:  f["tcp"]  += 1
    if p == 17: f["udp"]  += 1
    if p == 1:  f["icmp"] += 1
    fl = pkt["tcp_flags"]
    if fl & 0x02: f["syn"] += 1
    if fl & 0x01: f["fin"] += 1
    if fl & 0x04: f["rst"] += 1
    if fl & 0x10: f["ack"] += 1
    if fl & 0x08: f["psh"] += 1
    if fl & 0x80: f["cwr"] += 1
    if fl & 0x40: f["ece"] += 1
    f["src_mac"] = pkt["src_mac"]
    f["dst_mac"] = pkt["dst_mac"]
    return key

def compute_features(f):
    pkts = f["pkts"]
    n    = len(pkts)
    if n < 2: return None
    arr      = np.array(pkts, dtype=np.float64)
    tot_size = float(arr.sum())
    avg      = float(arr.mean())
    variance = float(arr.var())
    std      = float(arr.std())
    sum_sq   = float((arr ** 2).sum())
    magnitue = float(np.sqrt(sum_sq))
    radius   = float(np.sqrt(variance + sum_sq / n))
    dur      = max(float((f["last_ts"] or 0) - (f["first_ts"] or 0)), 0.001)
    weight   = n / dur
    cov      = float(sum((i - n/2.0) * (pkts[i] - avg) for i in range(n)) / n)
    iat_avg  = (f["iat_sum"] / f["iat_count"]) if f["iat_count"] > 0 else 0.0
    hdr_avg  = f["hdr_len"] / n
    tcp_r    = f["tcp"]  / n
    udp_r    = f["udp"]  / n
    icmp_r   = f["icmp"] / n
    if   f["tcp"]  >= f["udp"]  and f["tcp"]  >= f["icmp"]: proto_type = 6.0
    elif f["udp"]  >= f["tcp"]  and f["udp"]  >= f["icmp"]: proto_type = 17.0
    else:                                                     proto_type = 1.0

    feat = {
        "AVG": avg, "Covariance": cov, "Duration": dur,
        "Header_Length": hdr_avg, "IAT": iat_avg, "ICMP": icmp_r,
        "Magnitue": magnitue, "Max": float(arr.max()), "Min": float(arr.min()),
        "Number": float(n), "Protocol_Type": proto_type, "Radius": radius,
        "Std": std, "TCP": tcp_r, "Tot_size": tot_size, "UDP": udp_r,
        "Variance": variance, "Weight": weight,
        "ack_count": float(f["ack"]), "ack_flag_number": f["ack"]/n,
        "fin_count": float(f["fin"]), "psh_flag_number": f["psh"]/n,
        "rst_count": float(f["rst"]), "rst_flag_number": f["rst"]/n,
        "syn_count": float(f["syn"]), "syn_flag_number": f["syn"]/n,
    }

    import pandas as pd
    df = pd.DataFrame([feat], columns=MODEL_FEATURES)
    return df, feat

# ── Background flush + inference + MQTT ───────────────────────
def flush_flows():
    import pandas as pd
    while True:
        time.sleep(2)
        with lock:
            keys_to_process = list(flow_table.keys())

        for key in keys_to_process:
            with lock:
                if key not in flow_table: continue
                f = flow_table[key]
                if len(f["pkts"]) < 2: continue
                result = compute_features(f)
                if result is None: continue
                df, feat = result

            try:
                if scaler:
                    X = scaler.transform(df)
                    X = pd.DataFrame(X, columns=MODEL_FEATURES)
                else:
                    X = df

                pred   = int(model.predict(X)[0])
                proba  = model.predict_proba(X)[0]
                conf   = float(proba[pred]) * 100.0
                pname  = CLASS_NAMES.get(pred, "Unknown")
                threat = THREAT_LEVEL.get(pred, "NONE")
                proto  = {6:"TCP", 17:"UDP", 1:"ICMP"}.get(key[2], "?")
                now_s  = time.strftime("%H:%M:%S")

                with lock:
                    f_snap = flow_table.get(key, {})
                    event = {
                        "time": now_s, "prediction": pname,
                        "confidence": round(conf, 1), "threat": threat,
                        "src_ip": key[0], "dst_ip": key[1],
                        "src_mac": f_snap.get("src_mac","??"),
                        "dst_mac": f_snap.get("dst_mac","??"),
                        "protocol": proto,
                        "num_packets": len(f_snap.get("pkts",[])),
                        "duration_s": round(float((f_snap.get("last_ts") or 0) -
                                                  (f_snap.get("first_ts") or 0)), 3),
                        "syn_count": f_snap.get("syn",0),
                        "ack_count": f_snap.get("ack",0),
                        "fin_count": f_snap.get("fin",0),
                        "rst_count": f_snap.get("rst",0),
                        "tot_size":  round(feat["Tot_size"],0),
                        "avg_pkt":   round(feat["AVG"],1),
                    }
                    events.appendleft(event)
                    stats["total_flows"] += 1
                    if pname == "BenignTraffic":
                        stats["benign"] += 1
                    else:
                        stats["total_attacks"] += 1
                        if pname in stats: stats[pname] += 1
                        timeline.appendleft({
                            "time": now_s, "prediction": pname,
                            "confidence": round(conf,1),
                            "src_ip": key[0], "threat": threat,
                        })
                        print(f"[ATTACK] {pname} | {conf:.1f}% | "
                              f"{key[0]}→{key[1]} | {proto}")

                        # ── MQTT alert ← NEW ──────────────────
                        m.publish(MQTT_TOPIC, json.dumps({
                            "attack":     pname,
                            "threat":     threat,
                            "confidence": round(conf, 1),
                            "src_ip":     key[0],
                            "dst_ip":     key[1],
                            "protocol":   proto,
                            "time":       now_s
                        }))

            except Exception as e:
                print(f"[ERROR] Inference failed for {key}: {e}")

        with lock:
            flow_table.clear()

threading.Thread(target=flush_flows, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "no JSON"}), 400

    with lock:
        stats["packets_received"] += 1

    pkt = parse_hex_payload(
        data.get("hex_payload",""),
        data.get("type",""),
        data.get("src_mac","??"),
        data.get("dst_mac","??"),
    )

    if pkt:
        with lock:
            update_flow(pkt, time.time())
            stats["packets_processed"] += 1

    return jsonify({"status": "ok"}), 200

@app.route("/data")
def get_data():
    with lock:
        return jsonify({
            "stats":    dict(stats),
            "events":   list(events)[:60],
            "timeline": list(timeline)[:30],
            "updated":  time.strftime("%H:%M:%S"),
        })

# ── Test MQTT route ← NEW ─────────────────────────────────────
@app.route("/test_mqtt")
def test_mqtt():
    m.publish(MQTT_TOPIC, json.dumps({
        "attack":     "DDoS",
        "threat":     "CRITICAL",
        "confidence": 99.5,
        "src_ip":     "192.168.1.100",
        "dst_ip":     "192.168.1.1",
        "protocol":   "TCP",
        "time":       time.strftime("%H:%M:%S")
    }))
    return "MQTT test alert sent to ids/alerts!"

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

# ── Dashboard HTML (unchanged) ────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IoMT IDS Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090d;--bg2:#0d1117;--bg3:#111820;--bg4:#141c26;
  --border:#1c2a3a;--border2:#243040;
  --text:#c9d1d9;--muted:#556070;--dim:#3a4a5a;
  --accent:#00d4ff;--accent2:#0099bb;
  --green:#00e676;--red:#ff3d3d;--orange:#ff8800;
  --yellow:#ffd600;--purple:#bb86fc;--pink:#ff6b9d;
  --mono:'Share Tech Mono',monospace;--sans:'Inter',sans-serif;
}
html,body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;font-size:14px}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,212,255,.008) 3px,rgba(0,212,255,.008) 4px);pointer-events:none;z-index:9999}
header{background:linear-gradient(135deg,#080f1c 0%,#0a1525 60%,#060d18 100%);border-bottom:1px solid var(--border2);padding:0 32px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;}
.logo{display:flex;align-items:center;gap:12px}
.logo-hex{width:36px;height:36px;flex-shrink:0;background:linear-gradient(135deg,var(--accent),var(--accent2));clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:12px;color:#000;font-weight:700;}
.logo-text h1{font-family:var(--mono);font-size:15px;color:var(--accent);letter-spacing:3px}
.logo-text p{font-size:10px;color:var(--muted);letter-spacing:1.5px;margin-top:2px}
.hright{display:flex;align-items:center;gap:24px}
.live-badge{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--green)}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.7)}}
.clock{font-family:var(--mono);font-size:12px;color:var(--muted)}
main{padding:20px 28px;max-width:1700px;margin:0 auto}
.stat-row{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:18px}
.stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:14px 12px 10px;position:relative;overflow:hidden;transition:border-color .2s;}
.stat-card:hover{border-color:var(--border2)}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--c,var(--accent))}
.stat-card.c-red{--c:var(--red)}.stat-card.c-green{--c:var(--green)}
.stat-card.c-ddos{--c:var(--red)}.stat-card.c-dos{--c:var(--orange)}
.stat-card.c-mirai{--c:var(--purple)}.stat-card.c-recon{--c:var(--yellow)}
.stat-card.c-blue{--c:var(--accent)}
.stat-val{font-family:var(--mono);font-size:26px;color:var(--c,var(--accent));line-height:1;margin-bottom:4px}
.stat-lbl{font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase}
.grid-main{display:grid;grid-template-columns:1fr 300px;gap:16px;margin-bottom:16px}
.grid-bottom{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--border);background:var(--bg3);}
.panel-title{font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:2px;text-transform:uppercase}
.panel-badge{font-family:var(--mono);font-size:10px;color:var(--muted);padding:2px 8px;border:1px solid var(--border2);border-radius:3px}
.feed-wrap{overflow-x:auto;max-height:420px;overflow-y:auto}
.feed-wrap::-webkit-scrollbar{width:3px;height:3px}
.feed-wrap::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
table{width:100%;border-collapse:collapse;min-width:650px}
thead th{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);background:var(--bg3);position:sticky;top:0;z-index:1;white-space:nowrap;}
tbody tr{border-bottom:1px solid rgba(28,42,58,.5);animation:fadeIn .35s ease}
@keyframes fadeIn{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:none}}
tbody tr:hover{background:rgba(0,212,255,.025)}
tbody td{padding:7px 10px;font-size:11px;font-family:var(--mono);white-space:nowrap}
td.t-ip{color:var(--accent)}td.t-mac{color:var(--dim);font-size:10px}td.t-time{color:var(--muted)}td.t-proto{color:var(--yellow)}td.t-num{color:var(--text)}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:10px;font-family:var(--mono);font-weight:600;letter-spacing:.5px}
.badge-benign{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.25)}
.badge-ddos{background:rgba(255,61,61,.12);color:var(--red);border:1px solid rgba(255,61,61,.35)}
.badge-dos{background:rgba(255,136,0,.12);color:var(--orange);border:1px solid rgba(255,136,0,.35)}
.badge-mirai{background:rgba(187,134,252,.12);color:var(--purple);border:1px solid rgba(187,134,252,.35)}
.badge-recon{background:rgba(255,214,0,.1);color:var(--yellow);border:1px solid rgba(255,214,0,.3)}
.thr{font-size:9px;font-family:var(--mono);padding:1px 6px;border-radius:2px;letter-spacing:1px}
.thr-critical{color:var(--red);background:rgba(255,61,61,.1)}.thr-high{color:var(--orange);background:rgba(255,136,0,.1)}
.thr-medium{color:var(--yellow);background:rgba(255,214,0,.1)}.thr-none{color:var(--muted)}
.conf-bar{display:flex;align-items:center;gap:5px}
.conf-track{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden;min-width:40px}
.conf-fill{height:100%;border-radius:2px;transition:width .4s}
.conf-val{font-size:10px;min-width:32px;text-align:right;color:var(--text)}
.sidebar{display:flex;flex-direction:column;gap:14px}
.dist-body{padding:14px 16px}
.dist-row{margin-bottom:12px}.dist-row:last-child{margin-bottom:0}
.dist-label{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;margin-bottom:5px}
.dist-label span:last-child{color:var(--muted)}
.dist-track{height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.dist-fill{height:100%;border-radius:3px;transition:width .6s ease;min-width:2px}
.fill-ddos{background:linear-gradient(90deg,#ff3d3d,#ff7070)}.fill-dos{background:linear-gradient(90deg,#ff8800,#ffbb44)}
.fill-mirai{background:linear-gradient(90deg,#bb86fc,#d4a8ff)}.fill-recon{background:linear-gradient(90deg,#ffd600,#ffe066)}
.info-body{padding:12px 16px}
.info-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(28,42,58,.4);font-size:11px}
.info-row:last-child{border:none}
.info-key{color:var(--muted);font-family:var(--mono)}.info-val{color:var(--accent);font-family:var(--mono);text-align:right}
.timeline-body{max-height:260px;overflow-y:auto;padding:8px 14px}
.timeline-body::-webkit-scrollbar{width:3px}.timeline-body::-webkit-scrollbar-thumb{background:var(--border2)}
.tl-item{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid rgba(28,42,58,.5);animation:fadeIn .35s ease}
.tl-item:last-child{border:none}
.tl-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:3px}
.tl-dot-ddos{background:var(--red);box-shadow:0 0 6px var(--red)}.tl-dot-dos{background:var(--orange);box-shadow:0 0 6px var(--orange)}
.tl-dot-mirai{background:var(--purple);box-shadow:0 0 6px var(--purple)}.tl-dot-recon{background:var(--yellow);box-shadow:0 0 6px var(--yellow)}
.tl-content{flex:1}.tl-top{display:flex;justify-content:space-between;margin-bottom:2px}
.tl-name{font-family:var(--mono);font-size:12px;font-weight:600}.tl-time{font-family:var(--mono);font-size:10px;color:var(--muted)}
.tl-sub{font-family:var(--mono);font-size:10px;color:var(--muted)}
.summary-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
.sum-item{padding:14px 16px;border-right:1px solid var(--border);border-bottom:1px solid var(--border)}
.sum-item:nth-child(2n){border-right:none}.sum-item:nth-last-child(-n+2){border-bottom:none}
.sum-val{font-family:var(--mono);font-size:22px;margin-bottom:3px}
.sum-lbl{font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase}
.empty{padding:40px;text-align:center;color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:1px}
@media(max-width:1300px){.stat-row{grid-template-columns:repeat(4,1fr)}.grid-main{grid-template-columns:1fr}}
@media(max-width:800px){.stat-row{grid-template-columns:repeat(2,1fr)}.grid-bottom{grid-template-columns:1fr}main{padding:12px}}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-hex">IDS</div>
    <div class="logo-text">
      <h1>IoMT Network IDS</h1>
      <p>ESP32-S3 · LightGBM · IoMT 2024 · 26 Features · 5 Classes</p>
    </div>
  </div>
  <div class="hright">
    <div class="live-badge"><div class="pulse"></div>LIVE</div>
    <div class="clock" id="clk">--:--:--</div>
  </div>
</header>
<main>
  <div class="stat-row">
    <div class="stat-card c-blue"><div class="stat-val" id="sPkts">0</div><div class="stat-lbl">Total Packets</div></div>
    <div class="stat-card c-green"><div class="stat-val" id="sBen">0</div><div class="stat-lbl">Normal</div></div>
    <div class="stat-card c-red"><div class="stat-val" id="sAtk">0</div><div class="stat-lbl">Attack</div></div>
    <div class="stat-card c-ddos"><div class="stat-val" id="sDDoS">0</div><div class="stat-lbl">DDoS</div></div>
    <div class="stat-card c-dos"><div class="stat-val" id="sDoS">0</div><div class="stat-lbl">DoS</div></div>
    <div class="stat-card c-mirai"><div class="stat-val" id="sMirai">0</div><div class="stat-lbl">Mirai</div></div>
    <div class="stat-card c-recon"><div class="stat-val" id="sRecon">0</div><div class="stat-lbl">Recon</div></div>
  </div>
  <div class="grid-main">
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Live Flow Feed</span><span class="panel-badge" id="feedCount">0 flows</span></div>
      <div class="feed-wrap">
        <table>
          <thead><tr><th>Time</th><th>Src IP</th><th>Dst IP</th><th>Proto</th><th>Pkts</th><th>Dur</th><th>Prediction</th><th>Confidence</th><th>Threat</th></tr></thead>
          <tbody id="feedBody"><tr><td colspan="9" class="empty">Waiting for packets from ESP32...</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="sidebar">
      <div class="panel">
        <div class="panel-head"><span class="panel-title">Attack Distribution</span></div>
        <div class="dist-body">
          <div class="dist-row"><div class="dist-label"><span>DDoS</span><span id="dv-ddos">0</span></div><div class="dist-track"><div class="dist-fill fill-ddos" id="db-ddos" style="width:0%"></div></div></div>
          <div class="dist-row"><div class="dist-label"><span>DoS</span><span id="dv-dos">0</span></div><div class="dist-track"><div class="dist-fill fill-dos" id="db-dos" style="width:0%"></div></div></div>
          <div class="dist-row"><div class="dist-label"><span>Mirai</span><span id="dv-mirai">0</span></div><div class="dist-track"><div class="dist-fill fill-mirai" id="db-mirai" style="width:0%"></div></div></div>
          <div class="dist-row"><div class="dist-label"><span>Recon</span><span id="dv-recon">0</span></div><div class="dist-track"><div class="dist-fill fill-recon" id="db-recon" style="width:0%"></div></div></div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><span class="panel-title">System Info</span></div>
        <div class="info-body">
          <div class="info-row"><span class="info-key">Sensor</span><span class="info-val">ESP32-S3</span></div>
          <div class="info-row"><span class="info-key">Model</span><span class="info-val">LightGBM</span></div>
          <div class="info-row"><span class="info-key">Dataset</span><span class="info-val">IoMT 2024</span></div>
          <div class="info-row"><span class="info-key">Features</span><span class="info-val">26 flow</span></div>
          <div class="info-row"><span class="info-key">Classes</span><span class="info-val">5</span></div>
          <div class="info-row"><span class="info-key">Attack Rate</span><span class="info-val" id="iAtkRate">0.0%</span></div>
          <div class="info-row"><span class="info-key">Threat Level</span><span class="info-val" id="iThreat" style="color:var(--muted)">NONE</span></div>
          <div class="info-row"><span class="info-key">Updated</span><span class="info-val" id="iUpd">--:--:--</span></div>
        </div>
      </div>
    </div>
  </div>
  <div class="grid-bottom">
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Attack Timeline</span><span class="panel-badge" id="tlCount">0 events</span></div>
      <div class="timeline-body" id="tlBody"><div class="empty">No attacks detected yet</div></div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Detection Summary</span></div>
      <div class="summary-grid">
        <div class="sum-item"><div class="sum-val" id="sumFlows" style="color:var(--accent)">0</div><div class="sum-lbl">Total Flows</div></div>
        <div class="sum-item"><div class="sum-val" id="sumAtk" style="color:var(--red)">0</div><div class="sum-lbl">Attacks</div></div>
        <div class="sum-item"><div class="sum-val" id="sumBen" style="color:var(--green)">0</div><div class="sum-lbl">Benign</div></div>
        <div class="sum-item"><div class="sum-val" id="sumRecon" style="color:var(--yellow)">0</div><div class="sum-lbl">Recon</div></div>
        <div class="sum-item"><div class="sum-val" id="sumRate" style="color:var(--orange)">0.0%</div><div class="sum-lbl">Attack Rate</div></div>
        <div class="sum-item"><div class="sum-val" id="sumPkts" style="color:var(--muted)">0</div><div class="sum-lbl">Packets Rx</div></div>
      </div>
    </div>
  </div>
</main>
<script>
const el = id => document.getElementById(id);
function predClass(p){return{BenignTraffic:'benign',DDoS:'ddos',DoS:'dos',Mirai:'mirai',Recon:'recon'}[p]||'benign'}
function dotClass(p){return{DDoS:'tl-dot-ddos',DoS:'tl-dot-dos',Mirai:'tl-dot-mirai',Recon:'tl-dot-recon'}[p]||'tl-dot-ddos'}
function confColor(c){return c>=90?'#00e676':c>=70?'#ffd600':'#ff8800'}
function thrClass(t){return{CRITICAL:'thr-critical',HIGH:'thr-high',MEDIUM:'thr-medium',NONE:'thr-none'}[t]||'thr-none'}
function updateStats(s){
  el('sPkts').textContent=(s.packets_received||0).toLocaleString();el('sBen').textContent=(s.benign||0).toLocaleString();
  el('sAtk').textContent=(s.total_attacks||0).toLocaleString();el('sDDoS').textContent=(s.DDoS||0).toLocaleString();
  el('sDoS').textContent=(s.DoS||0).toLocaleString();el('sMirai').textContent=(s.Mirai||0).toLocaleString();
  el('sRecon').textContent=(s.Recon||0).toLocaleString();
  const total=Math.max(s.total_flows||1,1),atk=s.total_attacks||0,rate=((atk/total)*100).toFixed(1)+'%';
  el('iAtkRate').textContent=rate;el('sumRate').textContent=rate;el('sumFlows').textContent=(s.total_flows||0).toLocaleString();
  el('sumAtk').textContent=(s.total_attacks||0).toLocaleString();el('sumBen').textContent=(s.benign||0).toLocaleString();
  el('sumRecon').textContent=(s.Recon||0).toLocaleString();el('sumPkts').textContent=(s.packets_received||0).toLocaleString();
  const mx=Math.max(s.DDoS||0,s.DoS||0,s.Mirai||0,s.Recon||0,1);
  el('dv-ddos').textContent=s.DDoS||0;el('dv-dos').textContent=s.DoS||0;el('dv-mirai').textContent=s.Mirai||0;el('dv-recon').textContent=s.Recon||0;
  el('db-ddos').style.width=((s.DDoS||0)/mx*100)+'%';el('db-dos').style.width=((s.DoS||0)/mx*100)+'%';
  el('db-mirai').style.width=((s.Mirai||0)/mx*100)+'%';el('db-recon').style.width=((s.Recon||0)/mx*100)+'%';
  let thr='NONE',tc='var(--muted)';
  if((s.DDoS||0)>0||(s.Mirai||0)>0){thr='CRITICAL';tc='var(--red)'}
  else if((s.DoS||0)>0){thr='HIGH';tc='var(--orange)'}
  else if((s.Recon||0)>0){thr='MEDIUM';tc='var(--yellow)'}
  el('iThreat').textContent=thr;el('iThreat').style.color=tc;
}
function updateFeed(events){
  el('feedCount').textContent=events.length+' flows';
  if(!events.length){el('feedBody').innerHTML='<tr><td colspan="9" class="empty">Waiting for packets from ESP32...</td></tr>';return;}
  el('feedBody').innerHTML=events.map(e=>{
    const pc=predClass(e.prediction),c=e.confidence||0,tc=thrClass(e.threat);
    return`<tr><td class="t-time">${e.time}</td><td class="t-ip">${e.src_ip}</td><td class="t-ip">${e.dst_ip}</td><td class="t-proto">${e.protocol}</td><td class="t-num">${e.num_packets}</td><td class="t-num">${e.duration_s}s</td><td><span class="badge badge-${pc}">${e.prediction}</span></td><td><div class="conf-bar"><div class="conf-track"><div class="conf-fill" style="width:${c}%;background:${confColor(c)}"></div></div><span class="conf-val">${c}%</span></div></td><td><span class="thr ${tc}">${e.threat}</span></td></tr>`;
  }).join('');
}
function updateTimeline(tl){
  el('tlCount').textContent=tl.length+' events';
  if(!tl.length){el('tlBody').innerHTML='<div class="empty">No attacks detected yet</div>';return;}
  el('tlBody').innerHTML=tl.map(t=>`<div class="tl-item"><div class="tl-dot ${dotClass(t.prediction)}"></div><div class="tl-content"><div class="tl-top"><span class="tl-name" style="color:${t.prediction==='DDoS'?'var(--red)':t.prediction==='DoS'?'var(--orange)':t.prediction==='Mirai'?'var(--purple)':'var(--yellow)'}">${t.prediction}</span><span class="tl-time">${t.time}</span></div><div class="tl-sub">${t.src_ip} · ${t.confidence}% · ${t.threat}</div></div></div>`).join('');
}
async function poll(){
  try{const r=await fetch('/data');const d=await r.json();el('clk').textContent='Updated: '+d.updated;el('iUpd').textContent=d.updated;updateStats(d.stats);updateFeed(d.events);updateTimeline(d.timeline);}
  catch(e){el('clk').textContent='Connection lost...';}
}
poll();setInterval(poll,2000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"\n{'='*54}")
    print(f"  IoMT IDS Server + MQTT")
    print(f"  Dashboard : http://{local_ip}:5000")
    print(f"  Endpoint  : http://{local_ip}:5000/predict")
    print(f"  Test MQTT : http://{local_ip}:5000/test_mqtt")
    print(f"  MQTT      : ids/alerts -> HiveMQ")
    print(f"{'='*54}\n")

    threading.Thread(target=start_mdns, args=(local_ip,), daemon=True).start()
    threading.Thread(target=udp_beacon, args=(local_ip,), daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False)