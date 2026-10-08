"""Robust resumable downloader for BindingNet v1's Crystal_Templates archive (2,182,806,108 bytes, Zenodo
record 11069450). The previous attempt (data/bindingnet_v1/resume.log) stalled repeatedly around 50-105MB
over 22 tries, all logged as generic "HTTPError". Diagnosed directly (see chat): Zenodo actually serves the
Range request fine (HTTP 206, correct content-range/content-length) -- the response headers show a
`x-ratelimit-limit: 133` / `retry-after: 60` policy, so the old script's fast-retry-on-drop loop was almost
certainly getting rate-limited mid-transfer, not genuinely rejected. This version reads in bounded chunks
(default 25MB) via a fresh Range request each time, appends to the output file, and backs off using the
server's own `Retry-After` header when present (falling back to exponential backoff otherwise) -- so a
dropped connection costs one chunk, not the whole resume.

Usage:
  python guidance/surrogate_data/resume_crystal_templates.py
"""
import os
import time
import urllib.error
import urllib.request

URL = 'https://zenodo.org/api/records/11069450/files/Crystal_Templates_for_BindingNet1.tar.gz/content'
OUT = './data/bindingnet_v1/Crystal_Templates_for_BindingNet1.tar.gz'
LOG = './data/bindingnet_v1/resume.log'
TOTAL = 2182806108
CHUNK = 5 * 1024 * 1024


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def main():
    done = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    log(f'=== resume_crystal_templates.py starting at offset {done}/{TOTAL} ===')
    consecutive_failures = 0
    while done < TOTAL:
        end = min(done + CHUNK - 1, TOTAL - 1)
        req = urllib.request.Request(URL, headers={'Range': f'bytes={done}-{end}'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(OUT, 'ab') as f:
                f.write(data)
            done += len(data)
            consecutive_failures = 0
            pct = 100 * done / TOTAL
            log(f'{done}/{TOTAL} ({pct:.2f}%)')
        except urllib.error.HTTPError as e:
            consecutive_failures += 1
            retry_after = e.headers.get('Retry-After')
            wait = int(retry_after) if retry_after else min(60, 5 * consecutive_failures)
            log(f'HTTPError {e.code} at offset {done}, waiting {wait}s (failure #{consecutive_failures})')
            time.sleep(wait)
        except Exception as e:
            consecutive_failures += 1
            wait = min(60, 5 * consecutive_failures)
            log(f'{type(e).__name__}: {e} at offset {done}, waiting {wait}s (failure #{consecutive_failures})')
            time.sleep(wait)
        if consecutive_failures >= 30:
            log('30 consecutive failures, giving up for now -- rerun this script to resume from here.')
            return
    log('=== download complete ===')


if __name__ == '__main__':
    main()
