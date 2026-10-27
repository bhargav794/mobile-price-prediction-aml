"""Start the actual Streamlit server, verify HTTP health, and stop only our process."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    started = time.monotonic()
    log_path = ROOT / 'reports/server_smoke.log'
    with tempfile.TemporaryDirectory(prefix='aml-streamlit-') as workdir, log_path.open('w') as log:
        command = [sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
                   '--server.headless=true', '--server.address=127.0.0.1',
                   f'--server.port={port}', '--browser.gatherUsageStats=false']
        process = subprocess.Popen(command, cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONPATH': str(ROOT)})
        try:
            url = f'http://127.0.0.1:{port}'
            while time.monotonic() - started < 30:
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text())
                try:
                    with urllib.request.urlopen(url + '/_stcore/health', timeout=2) as response:
                        status = response.status
                        health = response.read().decode()
                    break
                except OSError:
                    time.sleep(.25)
            else:
                raise TimeoutError('Streamlit did not become healthy within 30 seconds.')
            with urllib.request.urlopen(url, timeout=5) as response:
                page_status = response.status
                page = response.read().decode()
            assert status == page_status == 200 and health.strip() == 'ok'
            assert '<html' in page.lower()
            result = {'health_status': status, 'health_body': health,
                      'page_status': page_status, 'started_from_other_working_directory': True,
                      'startup_seconds': round(time.monotonic()-started, 3),
                      'server_stopped_after_check': True, 'public_deployment': False}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    (ROOT / 'reports/server_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
