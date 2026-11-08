"""Check a public Streamlit app's HTTP health and rendered session messages."""
import json
import time
from pathlib import Path
import requests
from websockets.sync.client import connect
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

ROOT = Path(__file__).resolve().parent
BASE = 'https://mobile-price-prediction-aml.streamlit.app'


def main():
    session = requests.Session()
    response = session.get(BASE, timeout=25)
    response.raise_for_status()
    health = session.get(BASE + '/_stcore/health', timeout=25)
    health.raise_for_status()
    cookies = '; '.join(f'{c.name}={c.value}' for c in session.cookies)
    headers = {'Cookie': cookies} if cookies else {}
    messages = []
    finished = False
    limitation = None
    try:
        with connect(BASE.replace('https:', 'wss:') + '/_stcore/stream',
                     origin=BASE, subprotocols=['streamlit'], additional_headers=headers,
                     open_timeout=20, max_size=16*1024*1024) as websocket:
            rerun = BackMsg()
            rerun.rerun_script.query_string = ''
            websocket.send(rerun.SerializeToString())
            started = time.monotonic()
            while time.monotonic() - started < 45:
                raw = websocket.recv(timeout=15)
                if not isinstance(raw, bytes):
                    continue
                message = ForwardMsg()
                message.ParseFromString(raw)
                messages.append(str(message))
                if message.WhichOneof('type') == 'script_finished':
                    finished = True
                    break
    except Exception as exc:
        limitation = f'{type(exc).__name__}: {exc}'
    text = '\n'.join(messages)
    expected = ['Model and evaluation', 'Assumptions used', 'p_0, p_1, p_2 and p_3']
    removed = ['Its provenance', 'This is a predictive association, not a causal experiment',
               'Probability estimates are not a promise of correctness']
    result = {'url': BASE + '/', 'http_status': response.status_code,
              'health_status': health.status_code, 'health_body_is_ok':health.text.strip() == 'ok',
              'health_response_preview':health.text[:180],
              'session_finished': finished, 'rendered_text_verified': finished,
              'session_verification_limitation': limitation,
              'new_text_found': {x: x in text if finished else None for x in expected},
              'removed_text_absent': {x: x not in text if finished else None for x in removed},
              'session_message_count': len(messages)}
    (ROOT / 'reports/live_app_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if finished:
        assert all(result['new_text_found'].values()) and all(result['removed_text_absent'].values())


if __name__ == '__main__':
    main()
