"""验证双源降级：东财挂了自动切新浪"""
import requests

BASE = "http://localhost:8000/api"
r = requests.post(f"{BASE}/auth/login", data={"username": "testuser3", "password": "test1234"})
token = r.json().get("access_token", "")
headers = {"Authorization": f"Bearer {token}"}

# 测试 kline
print("=== kline 000001 ===")
r = requests.get(f"{BASE}/market/kline", params={
    "symbol": "000001",
    "period": "daily",
    "start_date": "20260401",
    "end_date": "20260424",
    "adjust": "qfq",
    "columns": "Close,Volume"
}, headers=headers)
print("status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print(f"name={data.get('name')} count={data.get('count')} cols={data.get('columns')}")
    if data.get("data"):
        d = data["data"][0]
        print(f"first: date={d['date']} close={d.get('close')} vol={d.get('volume')}")
else:
    print("error:", r.text[:300])

# 测试 search
print("\n=== search ===")
r = requests.get(f"{BASE}/market/search", params={"keyword": "600519", "limit": 3}, headers=headers)
print("status:", r.status_code)
if r.status_code == 200:
    for item in r.json().get("items", []):
        print(f"  {item['code']} - {item['name']}")
