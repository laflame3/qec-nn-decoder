"""Download a subset of members from the Zenodo 11403595 surface-code zip
without fetching the whole 6.2 GB archive (HTTP range requests).

Usage (run with -I, from the project root):
    ../../.venv/bin/python -I src/fetch_zenodo_subset.py plan  OUT_DIR
    ../../.venv/bin/python -I src/fetch_zenodo_subset.py fetch OUT_DIR

Selection rule: every file under a d3_* patch, plus the dataset README,
excluding measurements.b8 and sweep_bits.b8 (raw data not needed; detection
events and observable flips are provided separately).

Members are written to OUT_DIR keeping the archive's relative paths. Each
member's CRC-32 is checked against the zip central directory. Adjacent
members are fetched together in byte spans (gap < 16 MB, trading extra bytes for far fewer requests) to keep the number
of requests low. Already present files with a matching CRC are skipped, so
`fetch` can be re-run after an interruption.
"""
import concurrent.futures as cf
import io
import json
import os
import struct
import time
import sys
import urllib.error
import urllib.request
import zipfile
import zlib

URL = ("https://zenodo.org/records/11403595/files/"
       "google_sycamore_surface_code_d3_d5.zip?download=1")
UA = {"User-Agent": "qec-nn-decoder-research"}
EXCLUDE = {"measurements.b8", "sweep_bits.b8"}


def get(start, end, tries=8):
    """Range GET with backoff on HTTP 429 (Zenodo rate limit) and transient errors."""
    req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={start}-{end}"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 30 * (k + 1)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            wait = 10 * (k + 1)
        print(f"retry in {wait}s (bytes {start}-{end})", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on bytes {start}-{end}")


class RangeFile(io.RawIOBase):
    def __init__(self):
        req = urllib.request.Request(URL, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        data = get(self.pos, min(self.pos + len(b), self.size) - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def selected(info):
    n = info.filename
    if n.endswith("/"):
        return False
    parts = n.split("/")
    if len(parts) >= 3 and parts[1] == "README":
        return True
    return (len(parts) >= 4 and parts[2].startswith("d3_")
            and parts[-1] not in EXCLUDE)


MAX_GAP = 16_000_000      # merge members whose gap is below this (few requests; Zenodo rate-limits)
MAX_SPAN = 128_000_000    # bytes per HTTP request


def extract(info, buf, base):
    """Return decompressed bytes of `info` from a buffer starting at `base`."""
    off = info.header_offset - base
    if struct.unpack("<I", buf[off:off + 4])[0] != 0x04034B50:
        raise ValueError(f"bad local header for {info.filename}")
    nlen, xlen = struct.unpack("<HH", buf[off + 26:off + 30])
    start = off + 30 + nlen + xlen
    raw = buf[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        data = bytes(raw)
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC or len(data) != info.file_size:
        raise ValueError(f"CRC/size mismatch for {info.filename}")
    return data


def already_ok(info, out_dir):
    out = os.path.join(out_dir, info.filename)
    if not os.path.exists(out):
        return False
    with open(out, "rb") as fh:
        return zlib.crc32(fh.read()) == info.CRC


def fetch_span(span, out_dir):
    s, e, members = span
    buf = get(s, e - 1)
    for info in members:
        data = extract(info, buf, s)
        out = os.path.join(out_dir, info.filename)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".part", "wb") as fh:
            fh.write(data)
        os.replace(out + ".part", out)
    return len(members)


def make_spans(infos, sel):
    order = sorted(infos, key=lambda i: i.header_offset)
    nxt = {a.filename: b.header_offset for a, b in zip(order, order[1:])}
    spans = []
    for i in sorted(sel, key=lambda i: i.header_offset):
        s = i.header_offset
        e = nxt.get(i.filename, s + 30 + 2048 + i.compress_size)
        if spans and s - spans[-1][1] <= MAX_GAP and e - spans[-1][0] <= MAX_SPAN:
            spans[-1][1] = e
            spans[-1][2].append(i)
        else:
            spans.append([s, e, [i]])
    return spans


def main():
    mode, out_dir = sys.argv[1], sys.argv[2]
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(), buffer_size=1 << 20))
    sel = [i for i in z.infolist() if selected(i)]
    comp = sum(i.compress_size for i in sel)
    unc = sum(i.file_size for i in sel)
    print(f"selected files: {len(sel)}  download≈{comp/1e9:.3f} GB  "
          f"on disk≈{unc/1e9:.3f} GB")
    if mode == "plan":
        return
    os.makedirs(out_dir, exist_ok=True)
    todo = [i for i in sel if not already_ok(i, out_dir)]
    spans = make_spans(z.infolist(), todo)
    print(f"to fetch: {len(todo)} files in {len(spans)} requests", flush=True)
    done = 0
    errors = []
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(fetch_span, sp, out_dir) for sp in spans]
        for k, f in enumerate(cf.as_completed(futs), 1):
            try:
                done += f.result()
            except Exception as e:  # keep going, report at the end; re-run to retry
                errors.append(repr(e))
            if k % 5 == 0:
                print(f"requests {k}/{len(spans)}  files {done}", flush=True)
    manifest = {
        "source": "https://doi.org/10.5281/zenodo.11403595",
        "archive": "google_sycamore_surface_code_d3_d5.zip",
        "archive_md5": "c1205ecff28c51f8cc74c2de5ee73261",
        "selection": "d3_* patches + README, excluding measurements.b8, sweep_bits.b8",
        "files": len(sel), "uncompressed_bytes": unc, "errors": errors,
        "fetched_this_run": done,
    }
    with open(os.path.join(out_dir, "subset_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"done: {len(sel)} files, {len(errors)} errors")
    for e in errors[:20]:
        print(e)


if __name__ == "__main__":
    main()
