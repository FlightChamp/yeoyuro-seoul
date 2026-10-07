"""
scripts/v2/29_watch_map.py
==========================
기본 좌표 파일을 감시하다가 **엑셀에서 저장할 때마다** 자동으로 검사하고 미리보기를 바꾼다.
사용자 파일에는 아무것도 쓰지 않는다 (엑셀을 열어 둔 채로 계속 작업하면 된다).

미리보기는 로컬 웹서버(http://localhost:8765)로 띄운다.
- 페이지 전체를 새로고침하지 않고, 1초마다 상태만 확인해 새 결과가 나오면 **그림만 교체** → 확대·스크롤 위치 유지
- 그림을 끝까지 다 쓴 뒤 새 이름으로 교체 → 쓰는 도중의 그림이나 브라우저 캐시 때문에 예전 그림이 보이지 않음
- 그림 클릭: 화면 맞춤 ↔ 원본 크기(스크롤해서 자세히 보기) 전환

실행 (터미널을 하나 띄워 두고)
----
    python scripts/v2/29_watch_map.py
    (끝내려면 Ctrl+C)
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import shutil
import socketserver
import threading
import time
import webbrowser
from datetime import datetime
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.map_preview_extras import stamp, zoom_changes   # noqa: E402
spec = importlib.util.spec_from_file_location("chk29", ROOT / "scripts" / "v2" / "29_check_map_workbook.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)
LIVE = ROOT / "reports" / "figures" / "map_live"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>노선도 실시간 미리보기</title>
<style>
body{font-family:'Malgun Gothic',sans-serif;margin:0;background:#f3f4f6}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 12px;font-size:15px;z-index:2}
#msg{font-size:13px;color:#555;margin-top:4px}
#wrap{padding:8px;overflow:auto}
img{display:block;background:#fff;border:1px solid #ddd;cursor:zoom-in}
img.fit{width:100%;height:auto}
img.full{width:auto;max-width:none;cursor:zoom-out}
.flash{animation:f 1.2s}@keyframes f{from{background:#fff3c4}to{background:#fff}}
</style></head><body>
<div id="bar"><b id="st">검사 대기 중…</b> · <span id="when"></span> · 엑셀에서 저장(Ctrl+S)하면 자동으로 바뀝니다 · 그림 왼쪽 위 #번호·saved 시각으로 새 그림인지 확인 · 그림 클릭 = 확대/맞춤
<div id="msg"></div></div>
<div id="zwrap" style="display:none;padding:8px"><img id="zoom" style="max-width:100%;border:2px solid #DC2626;background:#fff"></div>
<div id="wrap"><img id="img" class="fit" alt="미리보기"></div>
<script>
let ver=-1;
const img=document.getElementById('img');
img.onclick=()=>{img.className=img.className==='fit'?'full':'fit'};
async function poll(){
  try{
    const r=await fetch('status.json?'+Date.now(),{cache:'no-store'});
    const s=await r.json();
    if(s.version!==ver){
      ver=s.version;
      if(s.image){const n=new Image();n.onload=()=>{img.src=n.src};n.src=s.image+'?'+s.version;}
      const zw=document.getElementById('zwrap');
      if(s.zoom){document.getElementById('zoom').src=s.zoom+'?'+s.version;zw.style.display='block';}else{zw.style.display='none';}
      document.getElementById('st').textContent=s.status;
      document.getElementById('when').textContent='마지막 검사 '+s.when+' (저장 '+s.saved+')';
      document.getElementById('msg').innerHTML=s.detail;
      const b=document.getElementById('bar');b.classList.remove('flash');void b.offsetWidth;b.classList.add('flash');
    }
  }catch(e){document.getElementById('st').textContent='감시 스크립트가 꺼져 있습니다 (터미널에서 다시 실행)';}
  setTimeout(poll,1000);
}
poll();
</script></body></html>"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def serve() -> int:
    for port in range(8765, 8776):
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), partial(Quiet, directory=str(LIVE)))
            srv.daemon_threads = True
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            return port
        except OSError:
            continue
    raise RuntimeError("8765~8775 포트를 모두 쓸 수 없습니다")


def publish(res: dict | None, version: int, saved: str, err: str = "", changes: list | None = None,
            prev_snap: dict | None = None) -> None:
    image, zoom = "", ""
    if res is not None and res.get("preview_ok", res["preview"].exists()):
        image = f"preview_{version}.png"
        tmp = LIVE / (image + ".tmp")
        shutil.copy(res["preview"], tmp)
        try:
            stamp(tmp, f"#{version}  saved {saved}")       # 이 그림이 방금 저장한 결과인지 눈으로 확인
        except Exception:                                   # noqa: BLE001
            pass
        tmp.replace(LIVE / image)                         # 다 쓴 뒤에 교체
        if prev_snap is not None:
            ztmp = LIVE / f"zoom_{version}.png"
            try:
                if zoom_changes(res["frames"], prev_snap["final"], res["snap"]["final"], ztmp):
                    zoom = ztmp.name
            except Exception as e:                          # noqa: BLE001
                print(f"    (확대 그림 생성 실패: {type(e).__name__})")
        for old in list(LIVE.glob("preview_*.png")) + list(LIVE.glob("zoom_*.png")):
            if old.name not in (image, zoom):
                old.unlink(missing_ok=True)
    if res is None:
        status, detail = ("⚠️ 읽기 실패 — 다시 저장하면 재시도합니다" if err else "검사 중…"), err
    else:
        parts = []
        if res["clash"]:
            parts.append("다른 역끼리 같은 좌표: " + " / ".join(f"({x}, {y}) {', '.join(ks)}" for x, y, ks in res["clash"]))
        if res["missing"]:
            parts.append("좌표 빈칸: " + ", ".join(res["missing"][:15]))
        if res["fails"] > 0:
            t = res["table"]
            parts.append("검증 실패: " + " / ".join(f"{r.check} ({r.detail})" for r in t[t.status == 'FAIL'].itertuples()))
        status = "✅ 통과" if res["ok"] and not res["clash"] else ("⚠️ 통과 (겹침 있음)" if res["ok"] else "❌ 확인 필요")
        if changes is not None:
            if changes:
                parts.insert(0, "<b>이번 저장에서 고친 칸</b><br>· " + "<br>· ".join(changes[:30]))
            else:
                parts.insert(0, "<b>저장은 감지했지만 좌표 관련 칸은 바뀌지 않았습니다.</b> (다른 칸을 고쳤거나 같은 값으로 저장) "
                                "역 위치 = Station_Visual_Nodes 의 x_px·y_px, 역명 = Station_Labels 의 label_x_px·label_y_px 또는 label_dx·label_dy")
        detail = "<br>".join(parts)
        if not image:
            status, detail = "❌ 미리보기 생성 실패", res.get("stderr", "")[-400:]
    st = {"version": version, "status": status, "detail": detail, "image": image, "zoom": zoom,
          "when": datetime.now().strftime("%H:%M:%S"), "saved": saved}
    tmp = LIVE / "status.json.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LIVE / "status.json")


def main() -> int:
    base = chk.BASE
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "index.html").write_text(PAGE, encoding="utf-8")
    publish(None, 0, "-")
    port = serve()
    url = f"http://localhost:{port}/index.html"
    webbrowser.open(url)
    print(f"감시 중: {base}\n미리보기: {url}  (브라우저가 안 열리면 주소창에 붙여 넣기)\n"
          f"엑셀에서 저장할 때마다 그림만 바뀝니다. 끝내려면 Ctrl+C")
    from yeoyuro_v2.map_workbook import diff
    last, version, prev = None, 0, None
    while True:
        try:
            m = base.stat().st_mtime
            if m != last:
                time.sleep(0.8)                            # 엑셀이 저장을 끝낼 시간
                last = base.stat().st_mtime
                saved = datetime.fromtimestamp(last).strftime("%H:%M:%S")
                version += 1
                t0 = time.time()
                try:
                    res = chk.check(base)
                except Exception as e:                     # noqa: BLE001
                    publish(None, version, saved, f"{type(e).__name__}: {e}")
                    print(f"[{datetime.now():%H:%M:%S}] 읽기 실패 ({type(e).__name__}) — 다시 저장하면 재시도합니다")
                    continue
                changes = diff(prev, res["snap"]) if prev is not None else None
                if not res.get("preview_ok", True):
                    print(f"    [!] 미리보기 그림 생성 실패 — 예전 그림을 내보내지 않았습니다. 오류: {res.get('stderr', '')[-300:]}")
                publish(res, version, saved, changes=changes, prev_snap=prev)
                prev = res["snap"]
                print(f"[{datetime.now():%H:%M:%S}] 저장 감지 ({saved}) → {'통과' if res['ok'] else '확인 필요'} · "
                      f"겹침 {len(res['clash'])}곳 · 빈칸 {len(res['missing'])}개 · 미리보기 #{version} ({time.time() - t0:.1f}초)")
                if changes is None:
                    print("    (처음 검사 — 지금 상태를 기준으로 기억합니다)")
                elif changes:
                    for c in changes[:15]:
                        print(f"    · {c}")
                    if len(changes) > 15:
                        print(f"    … 외 {len(changes) - 15}개")
                else:
                    print("    [!] 저장은 감지했지만 좌표 관련 칸은 바뀌지 않았습니다 (다른 칸을 고쳤거나 같은 값으로 저장)")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("감시 종료")
            return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
