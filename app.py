from flask import Flask, request
import datetime
import collections
import math
import time
import os
import requests
from collections import defaultdict

app = Flask(__name__)

# ======================
# 設定値（必要に応じて調整）
# ======================
R_EARTH = 6371000.0     # 地球半径[m]
GEO_PREC = 8            # Geohash 精度（≈38mメッシュ）
THRESH_M = 10.0         # 近接判定の閾値[m]
TIME_WINDOW = 60.0      # 直近何秒の測位を比較するか
RELAX_RATIO = 1.2       # 近似距離でのゆるい判定倍率（誤脱落回避）

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")  # Discord Webhook（任意）


# ======================
# ユーザー管理
# ======================
users = {}

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.location_histry = collections.deque(maxlen=20)  # 直近20件の位置履歴
        self.last_ts = 0.0
        self.last_gh = None

    def add_location(self, lat, lon, ts=None):
        if ts is None:
            ts = time.time()
        self.location_histry.append((lat, lon))
        self.last_ts = ts

    def latest_location(self):
        if self.location_histry:
            return self.location_histry[-1]
        return None


# ======================
# 2点間距離（正確：ハ―バサイン）
# ======================
def haversine(lat1, lon1, lat2, lon2):
    """2地点間の距離を計算（mを返す）"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R_EARTH * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ======================
# 高速化ユーティリティ：Geohash
# ======================
__BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
__NEIGHBORS = {
    'right':  {'even': "bc01fg45238967deuvhjyznpkmstqrwx"},
    'left':   {'even': "238967debc01fg45kmstqrwxuvhjyznp"},
    'top':    {'even': "p0r21436x8zb9dcf5h7kjnmqesgutwvy"},
    'bottom': {'even': "14365h7k9dcfesgujnmqp0r2twvyx8zb"}
}
__BORDERS = {
    'right':  {'even': "bcfguvyz"},
    'left':   {'even': "0145hjnp"},
    'top':    {'even': "prxz"},
    'bottom': {'even': "028b"}
}
def _neighbor(hashcode: str, direction: str) -> str:
    if not hashcode:
        return ""
    last = hashcode[-1]
    parent = hashcode[:-1]
    type_ = 'even' if (len(hashcode) % 2) == 0 else 'odd'

    # 偶奇に応じてマップを切替（緯度経度の分割順の違いに起因）
    if type_ == 'odd':
        if direction in ('right','left'):
            direction_map = {
                'right': __NEIGHBORS['top']['even'],
                'left':  __NEIGHBORS['bottom']['even']
            }
            border_map = {
                'right': __BORDERS['top']['even'],
                'left':  __BORDERS['bottom']['even']
            }
        else:
            direction_map = {
                'top':    __NEIGHBORS['right']['even'],
                'bottom': __NEIGHBORS['left']['even']
            }
            border_map = {
                'top':    __BORDERS['right']['even'],
                'bottom': __BORDERS['left']['even']
            }
    else:
        direction_map = {d: __NEIGHBORS[d]['even'] for d in __NEIGHBORS}
        border_map    = {d: __BORDERS[d]['even']    for d in __BORDERS}

    if last in border_map[direction]:
        parent = _neighbor(parent, direction)
    idx = __BASE32.find(last)
    return (parent + __BASE32[direction_map[direction].find(__BASE32[idx])]) if parent is not None else ""

def neighbors(gh: str):
    n  = _neighbor(gh, 'top')
    s  = _neighbor(gh, 'bottom')
    e  = _neighbor(gh, 'right')
    w  = _neighbor(gh, 'left')
    ne = _neighbor(n, 'right')
    nw = _neighbor(n, 'left')
    se = _neighbor(s, 'right')
    sw = _neighbor(s, 'left')
    return {n, s, e, w, ne, nw, se, sw}

def encode_geohash(lat: float, lon: float, precision: int = GEO_PREC) -> str:
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(__BASE32[ch])
            bit = 0
            ch = 0
    return ''.join(geohash)


# ======================
# 高速化ユーティリティ：BBox / 近似距離
# ======================
def equirectangular_m(lat1, lon1, lat2, lon2):
    """等距円筒近似距離[m]（短距離で高速・十分高精度）"""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return R_EARTH * math.sqrt(x*x + y*y)

def bbox_pass(lat1, lon1, lat2, lon2, thresh_m):
    """バウンディングボックス（緯度・経度の差が大きすぎるものを即除外）"""
    deg_per_m = 180.0 / math.pi / R_EARTH
    dlat_max = thresh_m * deg_per_m
    # 経度は緯度で縮む（高緯度ガード付）
    dlon_max = dlat_max / max(0.2, math.cos(math.radians(lat1)))
    return (abs(lat1 - lat2) <= dlat_max) and (abs(lon1 - lon2) <= dlon_max)

def near_with_stages(lat1, lon1, lat2, lon2, thresh_m=THRESH_M, relax_ratio=RELAX_RATIO):
    """1) BBox → 2) 等距円筒近似 → 3) Haversine（確定）"""
    if not bbox_pass(lat1, lon1, lat2, lon2, thresh_m):
        return False
    if equirectangular_m(lat1, lon1, lat2, lon2) > (thresh_m * relax_ratio):
        return False
    return haversine(lat1, lon1, lat2, lon2) <= thresh_m


# ======================
# Geohash インデックス
# ======================
geo_index = defaultdict(set)  # geohash -> set(user_id)

def upsert_user(users_dict, user_id, lat, lon, ts=None):
    if user_id not in users_dict:
        users_dict[user_id] = User(user_id)
    u = users_dict[user_id]
    # 旧 geohash から外す
    if u.last_gh is not None:
        geo_index[u.last_gh].discard(user_id)
    # 位置更新
    u.add_location(lat, lon, ts=ts if ts is not None else time.time())
    # 新 geohash に追加
    gh = encode_geohash(lat, lon, GEO_PREC)
    u.last_gh = gh
    geo_index[gh].add(user_id)

def check_proximity_geohash(users_dict, user_id, threshold=THRESH_M, time_window=TIME_WINDOW):
    """
    user_id の最新位置を基準に、同セル＋8近傍セルだけを候補にして
    BBox→近似→Haversine で近接を判定。[(相手ID, 距離m), ...] を返す。
    """
    now = time.time()
    u = users_dict.get(user_id)
    if not u or not u.latest_location():
        return []
    lat1, lon1 = u.latest_location()
    gh = u.last_gh or encode_geohash(lat1, lon1, GEO_PREC)

    cand_hashes = {gh} | neighbors(gh)
    candidates = set()
    for h in cand_hashes:
        candidates |= geo_index.get(h, set())

    hits = []
    for vid in candidates:
        if vid == user_id:
            continue
        v = users_dict.get(vid)
        if not v or not v.latest_location():
            continue
        if (now - v.last_ts) > time_window:
            continue
        lat2, lon2 = v.latest_location()
        if near_with_stages(lat1, lon1, lat2, lon2, thresh_m=threshold, relax_ratio=RELAX_RATIO):
            d = haversine(lat1, lon1, lat2, lon2)
            hits.append((vid, d))
    return hits


# ======================
# Discord 通知（任意）
# ======================
def notify_discord(user_id: str, hits):
    """
    hits: [(相手ID, 距離m), ...]
    """
    if not DISCORD_WEBHOOK_URL or not hits:
        return
    lines = [f"**🚩 すれ違い検知** for `{user_id}`"]
    for vid, dist_m in hits:
        lines.append(f"- 相手: `{vid}` / 距離: **{dist_m:.2f} m**")
    payload = {"content": "\n".join(lines)}
    try:
        requests.post(DISCORD_WEBHOOK_URL, data=payload, timeout=5)
    except Exception as e:
        print(f"[discord] notify error: {e}")


# ======================
# 互換用（従来の全探索：必要なら比較用）
# ======================
def check_proximity(users_dict, threshold=10):
    user_ids = list(users_dict.keys())
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            loc1 = users_dict[user_ids[i]].latest_location()
            loc2 = users_dict[user_ids[j]].latest_location()
            if loc1 and loc2:
                lat1, lon1 = loc1
                lat2, lon2 = loc2
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance <= threshold:
                    print(f"すれ違い成功: {user_ids[i]} と {user_ids[j]} が {distance:.2f}m")
                    return True
    return False


# ======================
# API: 位置受信（POST）
# ======================
@app.route('/api/location', methods=["POST"])
def location():
    data = request.get_json()
    user_id = data.get('userID')
    latitude = float(data.get('latitude'))
    longitude = float(data.get('longitude'))
    print(f"[{datetime.datetime.now()}], [userID: {user_id}] Lat: {latitude} Lon: {longitude}")

    # Geohashインデックスに反映
    upsert_user(users, user_id, latitude, longitude, ts=time.time())

    # 自分の周辺だけ高速探索（同セル＋8近傍 → BBox → 近似 → Haversine）
    hits = check_proximity_geohash(users, user_id, threshold=THRESH_M, time_window=TIME_WINDOW)
    for vid, dist_m in hits:
        print(f"すれ違い成功: {user_id} と {vid} が {dist_m:.2f}m")

    # Discord通知（任意設定時のみ）
    notify_discord(user_id, hits)

    return {
        "status": "OK",
        "hits": [{"user": vid, "distance_m": round(dist_m, 2)} for vid, dist_m in hits]
    }, 200


# ======================
# テスト用ページ（スマホ・iPadで簡単テスト）
# ======================
@app.route("/test")
def test_page():
    return """
<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Proximity Test</title>
<style>
  body{font-family:sans-serif;max-width:720px;margin:24px auto;padding:0 12px}
  input,button{font-size:16px;padding:8px;margin:6px 0}
  .log{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;background:#f7f7f7;padding:8px;border-radius:8px;max-height:40vh;overflow:auto}
</style>
<h1>Proximity Test</h1>
<p>User ID を入れて Start を押すと現在地を送信します。Stop でトラッキングを終了できます。</p>
<label>User ID: <input id="uid" placeholder="alice or bob"></label><br>
<button id="start">Start Tracking</button>
<button id="stop" disabled>Stop Tracking</button>
<div class="log" id="log"></div>
<script>
let watchId = null;
const log = (m)=>{ 
  const el=document.getElementById('log'); 
  el.textContent += m + "\\n"; 
  el.scrollTop = el.scrollHeight; 
};

async function postLocation(uid, lat, lon){
  const res = await fetch("/api/location",{
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ userID: uid, latitude: lat, longitude: lon })
  });
  const j = await res.json().catch(()=>({}));
  log("POST -> " + res.status + " " + JSON.stringify(j));
}

document.getElementById("start").onclick = ()=>{
  const uid = document.getElementById("uid").value.trim();
  if(!uid){ alert("User ID を入れてください"); return; }
  if(!('geolocation' in navigator)){
    alert("このブラウザはGeolocationに対応していません"); return;
  }
  log("Start watchPosition for " + uid);
  watchId = navigator.geolocation.watchPosition(
    (pos)=>{
      const {latitude, longitude, accuracy} = pos.coords;
      log(`loc: lat=${latitude.toFixed(6)}, lon=${longitude.toFixed(6)}, acc=${Math.round(accuracy)}m`);
      postLocation(uid, latitude, longitude).catch(e=>log("ERR post: " + e));
    },
    (err)=>{ log("geo error: " + err.message); },
    { enableHighAccuracy: true, maximumAge: 3000, timeout: 10000 }
  );
  document.getElementById("start").disabled = true;
  document.getElementById("stop").disabled = false;
};

document.getElementById("stop").onclick = ()=>{
  if(watchId !== null){
    navigator.geolocation.clearWatch(watchId);
    log("Stopped tracking");
    watchId = null;
    document.getElementById("start").disabled = false;
    document.getElementById("stop").disabled = true;
  }
};
</script>
"""

# ======================
# エントリポイント
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True, ssl_context=("cert.pem","key.pem"))

