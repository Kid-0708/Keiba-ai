"""公式サイトへの丁寧なアクセス（間隔をあける・robots.txtを守る・同じページは二度取らない）"""
import hashlib, os, time, urllib.robotparser
from urllib.parse import urlparse
import requests

UA = "keiba-ai personal research bot (low frequency; contact via GitHub repository)"
DELAY = float(os.environ.get("FETCH_DELAY", "3.0"))      # 1アクセスごとに3秒以上あける
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
_robots, _last = {}, [0.0]


def allowed(url):
    host = urlparse(url)
    base = f"{host.scheme}://{host.netloc}"
    if base not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = _session.get(base + "/robots.txt", timeout=20)
            if r.status_code == 200: rp.parse(r.text.splitlines())
            elif r.status_code == 404: rp.parse([])          # robots.txt が無い＝制限なし
            else: rp.disallow_all = True                       # 確認できないときは取得しない（安全側）
        except requests.RequestException:
            rp.disallow_all = True
        _robots[base] = rp
    return _robots[base].can_fetch(UA, url)


def fetch(url, data=None, cache=True, encoding=None):
    """GET（data指定時はPOST）。robots.txtで禁止されたURLは取得しない。結果はキャッシュする。"""
    if not allowed(url):
        raise PermissionError(f"robots.txt で許可されていないURLです: {url}")
    key = hashlib.sha1((url + repr(sorted((data or {}).items()))).encode()).hexdigest()
    path = os.path.join(CACHE, key[:2], key + ".html")
    if cache and os.path.exists(path):
        return open(path, encoding="utf-8").read()
    rp = _robots[f"{urlparse(url).scheme}://{urlparse(url).netloc}"]
    wait = max(DELAY, float(rp.crawl_delay(UA) or 0)) - (time.time() - _last[0])   # Crawl-delay も守る
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        try:
            r = _session.post(url, data=data, timeout=30) if data else _session.get(url, timeout=30)
            _last[0] = time.time()
            if r.status_code == 200:
                break
            if r.status_code in (403, 429):
                raise RuntimeError(f"アクセスを拒否されました（{r.status_code}）。取得を中止します: {url}")
        except requests.RequestException:
            pass
        time.sleep(DELAY * (attempt + 2))
    else:
        raise RuntimeError(f"取得に失敗しました: {url}")
    r.encoding = encoding or r.apparent_encoding
    text = r.text
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").write(text)
    return text
