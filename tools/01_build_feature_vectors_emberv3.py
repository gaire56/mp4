#!/usr/bin/env python3
"""
01_build_feature_vectors_emberv3.py

Build EMBERv3/THREMBERv3-compatible tabular feature vectors (dim=2568) from EMBER2024_CORE PE JSONL.

Supports:
- sharded mode: one .npy set per JSONL file (recommended for huge datasets)
- single mode: one big .npy per split

Outputs are clearly named using:
  {dataset_tag}__{jsonl_basename}__X_tab_emberv3.npy, etc.
"""

import argparse
import json
import os
import sys
import time
import subprocess
import traceback
from datetime import datetime
from glob import glob
from typing import Dict, Any, List

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.feature_extraction import FeatureHasher


# ----------------------------
# THREMBER v3 feature layout (dim = 2568)
# ----------------------------
# general(7) + histogram(256) + byteentropy(256) + strings(177) + header(74) +
# section(224) + imports(1282) + exports(129) + datadirectories(34) +
# richheader(33) + authenticode(8) + pefilewarnings(88) = 2568
FEATURE_DIM = 2568


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def wc_l(path: str) -> int:
    """Fast line count using wc -l, fallback to Python if wc fails."""
    try:
        out = subprocess.check_output(["wc", "-l", path], text=True).strip()
        return int(out.split()[0])
    except Exception:
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        return n


def list_jsonl(folder_or_file: str, pattern: str, recursive: bool = False) -> List[str]:
    """Return sorted list of jsonl files. Works for a folder or a single file."""
    if os.path.isfile(folder_or_file):
        return [folder_or_file]
    if not os.path.isdir(folder_or_file):
        raise FileNotFoundError(f"Not found: {folder_or_file}")

    if recursive:
        files = glob(os.path.join(folder_or_file, "**", pattern), recursive=True)
    else:
        files = glob(os.path.join(folder_or_file, pattern))

    files = sorted([p for p in files if p.endswith(".jsonl") and os.path.isfile(p)])
    if not files:
        raise RuntimeError(f"No .jsonl files found in {folder_or_file} with pattern {pattern}")
    return files


def load_warning_patterns_from_file(path: str) -> List[str]:
    pats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pats.append(line)
    return pats


def build_warning_matchers(patterns: List[str]):
    """
    THREMBER warning patterns are like:
      "Something..." (prefix match)
      "...something" (suffix match)
    """
    prefixes = set()
    suffixes = set()
    ids = {}
    for i, line in enumerate(patterns):
        if line.startswith("..."):
            suffixes.add(line[3:])
            ids[line] = i
        else:
            # THREMBER uses prefix = line[:-3] for patterns that end with "..."
            prefixes.add(line[:-3])
            ids[line] = i
    return prefixes, suffixes, ids


def _print_first_error_once(state: Dict[str, Any], msg: str):
    """Print only once per run (first error)."""
    if state.get("printed_first_error", False):
        return
    state["printed_first_error"] = True
    print("\n" + "=" * 90, file=sys.stderr)
    print("FIRST ERROR (printed once)", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 90 + "\n", file=sys.stderr)


# ----------------------------
# THREMBER v3 compatible vectorizer
# ----------------------------
class EmberV3Vectorizer:
    def __init__(self, strict_exports_bug: bool = True, warnings_file: str = ""):
        self.strict_exports_bug = strict_exports_bug

        if not warnings_file or not os.path.exists(warnings_file):
            raise RuntimeError(
                "warnings_file is required for reproducibility.\n"
                "Provide THREMBER src/thrember/pefile_warnings.txt via --warnings_file"
            )

        patterns = load_warning_patterns_from_file(warnings_file)
        if len(patterns) != 87:
            raise RuntimeError(
                f"pefile_warnings.txt must have exactly 87 non-empty lines, but got {len(patterns)}.\n"
                "Use the THREMBER file: src/thrember/pefile_warnings.txt"
            )

        self.warning_prefixes, self.warning_suffixes, self.warning_ids = build_warning_matchers(patterns)

        # Header enums (copied from THREMBER src/thrember/features.py)
        self._machine_types = [
            "IMAGE_FILE_MACHINE_UNKNOWN", "IMAGE_FILE_MACHINE_AM33", "IMAGE_FILE_MACHINE_AMD64",
            "IMAGE_FILE_MACHINE_ARM", "IMAGE_FILE_MACHINE_ARM64", "IMAGE_FILE_MACHINE_ARMNT",
            "IMAGE_FILE_MACHINE_EBC", "IMAGE_FILE_MACHINE_I386", "IMAGE_FILE_MACHINE_IA64",
            "IMAGE_FILE_MACHINE_M32R", "IMAGE_FILE_MACHINE_MIPS16", "IMAGE_FILE_MACHINE_MIPSFPU",
            "IMAGE_FILE_MACHINE_MIPSFPU16", "IMAGE_FILE_MACHINE_POWERPC", "IMAGE_FILE_MACHINE_POWERPCFP",
            "IMAGE_FILE_MACHINE_R4000", "IMAGE_FILE_MACHINE_SH3", "IMAGE_FILE_MACHINE_SH3DSP",
            "IMAGE_FILE_MACHINE_SH4", "IMAGE_FILE_MACHINE_SH5", "IMAGE_FILE_MACHINE_THUMB",
            "IMAGE_FILE_MACHINE_WCEMIPSV2",
        ]
        self._subsystem_types = [
            "IMAGE_SUBSYSTEM_UNKNOWN", "IMAGE_SUBSYSTEM_NATIVE", "IMAGE_SUBSYSTEM_WINDOWS_GUI",
            "IMAGE_SUBSYSTEM_WINDOWS_CUI", "IMAGE_SUBSYSTEM_OS2_CUI", "IMAGE_SUBSYSTEM_POSIX_CUI",
            "IMAGE_SUBSYSTEM_NATIVE_WINDOWS", "IMAGE_SUBSYSTEM_WINDOWS_CE_GUI", "IMAGE_SUBSYSTEM_EFI_APPLICATION",
            "IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER", "IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER", "IMAGE_SUBSYSTEM_EFI_ROM",
            "IMAGE_SUBSYSTEM_XBOX", "IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION",
        ]
        self._machine_types_dict = {k: v for v, k in enumerate(self._machine_types)}
        self._subsystem_types_dict = {k: v for v, k in enumerate(self._subsystem_types)}

        self._image_characteristics = [
            "RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED",
            "AGGRESIVE_WS_TRIM", "LARGE_ADDRESS_AWARE", "16BIT_MACHINE", "BYTES_REVERSED_LO",
            "32BIT_MACHINE", "DEBUG_STRIPPED", "REMOVABLE_RUN_FROM_SWAP", "NET_RUN_FROM_SWAP",
            "SYSTEM", "DLL", "UP_SYSTEM_ONLY", "BYTES_REVERSED_HI",
        ]
        self._dll_characteristics = [
            "HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT", "NO_ISOLATION",
            "NO_SEH", "NO_BIND", "APPCONTAINER", "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE",
        ]
        self._dos_members = [
            "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr", "e_minalloc", "e_maxalloc",
            "e_ss", "e_sp", "e_csum", "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid",
            "e_oeminfo", "e_lfanew",
        ]

        # IMPORTANT FIX:
        # THREMBER uses 77 string_count keys, and the index order is: sorted(regex_keys)
        self.string_count_keys_sorted = sorted([
            ".click(", "/bin/", "/dev/", "/EmbeddedFile", "/FlateDecode", "/URI", "/proc/", "/tmp/",
    "/usr/", "CreateRemoteThread", "crypt32.dll", "digitalocean", "domain", "dos_msg", "dropped",
    "email_addr", "esx", "ftp", "GetProcAddress", "GetSystemDefaultLangID", "http", "http://",
    "https://", "html", "id_rsa", "IE USER", "inet_addr", "Invoke-Command", "kernel32.dll",
    "KERNEL32.DLL", "LoadLibrary", "mscoree.dll", "mutex", "NoCerts", "ntdll.dll", "onlick",
    "Powershell", "powershell", "process", "python", "QueryPerformanceCounter", "Registry",
    "Request", "regsvr32", "RtlDecompressBuffer", "self_signed", "service", "socket", "ssh",
    "SYSTEM32", "url", "USER32.DLL", "VMWARE", "wininet.dll", "winlogon.exe",
    "ws2_32.dll", "WSASocket", "WScript", "x509", "X509", "zipfile", "zlib",
    "cryptbase.dll", "bcrypt.dll", "ntoskrnl.exe", "kernelbase.dll", "advapi32.dll",
    "user32.dll", "gdi32.dll", "shell32.dll", "rpcrt4.dll",
    ".exe", ".dll", "cmd.exe", "powershell.exe", "svchost.exe", "rundll32.exe"
        ])
        # sanity: must be 77
        if len(self.string_count_keys_sorted) != 77:
            raise RuntimeError(f"Internal error: expected 77 string_count keys, got {len(self.string_count_keys_sorted)}")
        self.regex_idxs = {k: i for i, k in enumerate(self.string_count_keys_sorted)}

    def _safe_list(self, x, n=None):
        if not isinstance(x, list):
            return []
        if n is None:
            return x
        return (x + [0] * n)[:n]

    # ---- Feature groups ----
    def general(self, raw: Dict[str, Any]) -> np.ndarray:
        g = raw.get("general", {}) or {}
        sb = self._safe_list(g.get("start_bytes", [0, 0, 0, 0]), 4)
        return np.array([g.get("size", 0), g.get("entropy", 0), g.get("is_pe", 0), *sb], dtype=np.float32)

    def histogram(self, raw: Dict[str, Any]) -> np.ndarray:
        arr = np.array(raw.get("histogram", []) or [], dtype=np.float32)
        if arr.size != 256:
            arr = np.pad(arr[:256], (0, max(0, 256 - arr.size)), constant_values=0).astype(np.float32)
        s = float(arr.sum())
        return arr / (s if s > 0 else 1.0)

    def byteentropy(self, raw: Dict[str, Any]) -> np.ndarray:
        arr = np.array(raw.get("byteentropy", []) or [], dtype=np.float32)
        if arr.size != 256:
            arr = np.pad(arr[:256], (0, max(0, 256 - arr.size)), constant_values=0).astype(np.float32)
        s = float(arr.sum())
        return arr / (s if s > 0 else 1.0)

    def strings(self, raw: Dict[str, Any]) -> np.ndarray:
        s = raw.get("strings", {}) or {}
        printables = float(s.get("printables", 0) or 0)
        hist_div = printables if printables > 0 else 1.0

        pd = np.array(s.get("printabledist", []) or [], dtype=np.float32)
        if pd.size != 96:
            pd = np.pad(pd[:96], (0, max(0, 96 - pd.size)), constant_values=0).astype(np.float32)

        counts = np.zeros(77, dtype=np.float32)
        sc = s.get("string_counts", {}) or {}
        for k, v in sc.items():
            idx = self.regex_idxs.get(k)
            if idx is not None:
                counts[idx] = float(v)

        # 3 scalars + 96 + 1 scalar + 77 = 177
        return np.hstack([
            float(s.get("numstrings", 0)),
            float(s.get("avlength", 0)),
            float(s.get("printables", 0)),
            pd / hist_div,
            float(s.get("entropy", 0)),
            counts,
        ]).astype(np.float32)

    def header(self, raw: Dict[str, Any]) -> np.ndarray:
        h = raw.get("header", {}) or {}
        if not h:
            return np.zeros(74, dtype=np.float32)

        coff = h.get("coff", {}) or {}
        opt = h.get("optional", {}) or {}
        dos = h.get("dos", {}) or {}

        coff_chars = set(coff.get("characteristics", []) or [])
        dll_chars = set(opt.get("dll_characteristics", []) or [])

        vec = [
            float(coff.get("timestamp", 0)),
            float(coff.get("number_of_sections", 0)),
            float(coff.get("number_of_symbols", 0)),
            float(coff.get("sizeof_optional_header", 0)),
            float(coff.get("pointer_to_symbol_table", 0)),
            float(self._machine_types_dict.get(coff.get("machine", "IMAGE_FILE_MACHINE_UNKNOWN"), 0)),
            float(self._subsystem_types_dict.get(opt.get("subsystem", "IMAGE_SUBSYSTEM_UNKNOWN"), 0)),
            float(opt.get("major_image_version", 0)),
            float(opt.get("minor_image_version", 0)),
            float(opt.get("major_linker_version", 0)),
            float(opt.get("minor_linker_version", 0)),
            float(opt.get("major_operating_system_version", 0)),
            float(opt.get("minor_operating_system_version", 0)),
            float(opt.get("major_subsystem_version", 0)),
            float(opt.get("minor_subsystem_version", 0)),
            float(opt.get("sizeof_code", 0)),
            float(opt.get("sizeof_headers", 0)),
            float(opt.get("sizeof_image", 0)),
            float(opt.get("sizeof_initialized_data", 0)),
            float(opt.get("sizeof_uninitialized_data", 0)),
            float(opt.get("sizeof_stack_reserve", 0)),
            float(opt.get("sizeof_stack_commit", 0)),
            float(opt.get("sizeof_heap_reserve", 0)),
            float(opt.get("sizeof_heap_commit", 0)),
            float(opt.get("address_of_entrypoint", 0)),
            float(opt.get("base_of_code", 0)),
            float(opt.get("image_base", 0)),
            float(opt.get("section_alignment", 0)),
            float(opt.get("checksum", 0)),
            float(opt.get("number_of_rvas_and_sizes", 0)),
        ]
        vec += [1.0 if ch in coff_chars else 0.0 for ch in self._image_characteristics]
        vec += [1.0 if ch in dll_chars else 0.0 for ch in self._dll_characteristics]
        vec += [float(dos.get(m, 0)) for m in self._dos_members]

        out = np.array(vec, dtype=np.float32)
        if out.size != 74:
            out = np.pad(out[:74], (0, max(0, 74 - out.size)), constant_values=0).astype(np.float32)
        return out

    def section(self, raw: Dict[str, Any]) -> np.ndarray:
        s = raw.get("section", {}) or {}
        if not s:
            return np.zeros(224, dtype=np.float32)

        sections = s.get("sections", []) or []
        overlay = s.get("overlay", {}) or {"size": 0, "size_ratio": 0, "entropy": 0}
        entry = s.get("entry", "") or ""

        n_sections = len(sections)
        n_zero = sum(1 for sec in sections if (sec.get("size", 0) or 0) == 0)
        n_empty_name = sum(1 for sec in sections if (sec.get("name", "") or "") == "")
        n_rx = sum(
            1 for sec in sections
            if ("MEM_READ" in (sec.get("props", []) or []) and "MEM_EXECUTE" in (sec.get("props", []) or []))
        )
        n_w = sum(1 for sec in sections if ("MEM_WRITE" in (sec.get("props", []) or [])))

        entropies = [float(sec.get("entropy", 0)) for sec in sections] + [float(overlay.get("entropy", 0)), 0.0]
        size_ratios = [float(sec.get("size_ratio", 0)) for sec in sections] + [float(overlay.get("size_ratio", 0)), 0.0]
        vsize_ratios = [float(sec.get("vsize_ratio", 0)) for sec in sections] + [0.0]

        general = np.asarray([
            n_sections, n_zero, n_empty_name, n_rx, n_w,
            max(entropies), min(entropies),
            max(size_ratios), min(size_ratios),
            max(vsize_ratios), min(vsize_ratios),
        ], dtype=np.float32)

        section_sizes = [(sec.get("name", ""), float(sec.get("size", 0))) for sec in sections]
        section_vsize = [(sec.get("name", ""), float(sec.get("vsize", 0))) for sec in sections]
        section_entropy = [(sec.get("name", ""), float(sec.get("entropy", 0))) for sec in sections]
        characteristics = [f"{sec.get('name', '')}:{p}" for sec in sections for p in (sec.get("props", []) or [])]

        hs_sizes = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        hs_vsize = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        hs_entropy = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        hs_chars = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]
        hs_entry = FeatureHasher(10, input_type="string").transform([[entry]]).toarray()[0]

        out = np.hstack([
            general,
            hs_sizes, hs_vsize, hs_entropy, hs_chars, hs_entry,
            float(overlay.get("size", 0)),
            float(overlay.get("size_ratio", 0)),
            float(overlay.get("entropy", 0)),
        ]).astype(np.float32)

        if out.size != 224:
            out = np.pad(out[:224], (0, max(0, 224 - out.size)), constant_values=0).astype(np.float32)
        return out

    def imports(self, raw: Dict[str, Any]) -> np.ndarray:
        im = raw.get("imports", {}) or {}
        if not isinstance(im, dict) or not im:
            return np.zeros(1282, dtype=np.float32)

        libraries = list(set([str(l).lower() for l in im.keys()]))
        libs_h = FeatureHasher(256, input_type="string", alternate_sign=False).transform([libraries]).toarray()[0]

        imports = []
        for lib, funcs in im.items():
            lib_l = str(lib).lower()
            if isinstance(funcs, list):
                for fn in funcs:
                    imports.append(lib_l + ":" + str(fn))
        im_h = FeatureHasher(1024, input_type="string", alternate_sign=False).transform([imports]).toarray()[0]

        lengths = np.asarray([len(imports), len(libraries)], dtype=np.float32)
        return np.hstack([lengths, libs_h, im_h]).astype(np.float32)

    def exports(self, raw: Dict[str, Any]) -> np.ndarray:
        ex = raw.get("exports", []) or []
        if not isinstance(ex, list) or not ex:
            return np.zeros(129, dtype=np.float32)

        ex_list = [str(x) for x in ex]
        ex_h = FeatureHasher(128, input_type="string").transform([ex_list]).toarray()[0].astype(np.float32)

        if self.strict_exports_bug:
            count = float(len(ex_h))  # 128 (THREMBER behavior)
        else:
            count = float(len(ex_list))

        return np.hstack([np.asarray([count], dtype=np.float32), ex_h]).astype(np.float32)

    def datadirectories(self, raw: Dict[str, Any]) -> np.ndarray:
        dd = raw.get("datadirectories", []) or []
        if not isinstance(dd, list) or not dd:
            return np.zeros(34, dtype=np.float32)

        name_order = [
            "EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY", "BASERELOC", "DEBUG", "COPYRIGHT",
            "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT", "COM_DESCRIPTOR", "RESERVED"
        ]
        feat = np.zeros(2 * len(name_order) + 2, dtype=np.float32)

        try:
            # match THREMBER: for i in range(1, len(raw_obj)-1)
            for i in range(1, max(1, len(dd) - 1)):
                if not isinstance(dd[i], dict):
                    continue
                nm = dd[i].get("name", "")
                if nm in name_order:
                    idx = name_order.index(nm)
                    feat[2 * idx] = float(dd[i].get("size", 0))
                    feat[2 * idx + 1] = float(dd[i].get("virtual_address", 0))

            if isinstance(dd[0], dict):
                feat[-2] = float(dd[0].get("has_relocs", 0))
                feat[-1] = float(dd[0].get("has_dynamic_relocs", 0))
        except Exception:
            pass

        return feat

    def richheader(self, raw: Dict[str, Any]) -> np.ndarray:
        rh = raw.get("richheader", []) or []
        if not isinstance(rh, list) or not rh:
            return np.zeros(33, dtype=np.float32)

        n_pairs = int(len(rh) / 2)
        pairs = [(str(rh[i]), float(rh[i + 1])) for i in range(0, len(rh) - 1, 2)]
        h = FeatureHasher(32, input_type="pair").transform([pairs]).toarray()[0]
        return np.hstack([float(n_pairs), h]).astype(np.float32)

    def authenticode(self, raw: Dict[str, Any]) -> np.ndarray:
        au = raw.get("authenticode", {}) or {}
        if not isinstance(au, dict) or not au:
            return np.zeros(8, dtype=np.float32)
        keys = [
            "num_certs", "self_signed", "empty_program_name", "no_countersigner", "parse_error",
            "chain_max_depth", "latest_signing_time", "signing_time_diff"
        ]
        return np.asarray([float(au.get(k, 0)) for k in keys], dtype=np.float32)

    def pefilewarnings(self, raw: Dict[str, Any]) -> np.ndarray:
        w = raw.get("pefilewarnings", []) or []
        ids = np.zeros(88, dtype=np.float32)  # 87 flags + last dim=count
        if not isinstance(w, list) or not w:
            return ids

        for warn in w:
            # fast path: exact match
            if warn in self.warning_ids:
                ids[self.warning_ids[warn]] = 1.0
                continue

            s = str(warn)
            matched = False

            # suffix patterns "...XYZ"
            for suf in self.warning_suffixes:
                if s.endswith(suf):
                    key = "..." + suf
                    idx = self.warning_ids.get(key)
                    if idx is not None:
                        ids[idx] = 1.0
                    matched = True
                    break
            if matched:
                continue

            # prefix patterns "XYZ..."
            for pre in self.warning_prefixes:
                if s.startswith(pre):
                    key = pre + "..."
                    idx = self.warning_ids.get(key)
                    if idx is not None:
                        ids[idx] = 1.0
                    break

        ids[-1] = float(len(w))
        return ids

    def transform(self, obj: Dict[str, Any]) -> np.ndarray:
        v = np.hstack([
            self.general(obj),
            self.histogram(obj),
            self.byteentropy(obj),
            self.strings(obj),
            self.header(obj),
            self.section(obj),
            self.imports(obj),
            self.exports(obj),
            self.datadirectories(obj),
            self.richheader(obj),
            self.authenticode(obj),
            self.pefilewarnings(obj),
        ]).astype(np.float32)

        if v.size != FEATURE_DIM:
            raise RuntimeError(f"Vector dim mismatch: got {v.size}, expected {FEATURE_DIM}")
        return v


def smoke_test_jsonl(fp: str, vec: EmberV3Vectorizer, label_key: str, id_key: str):
    """Fail fast before writing huge memmaps if vectorization is broken."""
    with open(fp, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            _ = int(obj.get(label_key, 0))
            _ = str(obj.get(id_key, ""))
            _ = vec.transform(obj)
            return
    raise RuntimeError(f"Smoke test failed: no usable lines in {fp}")


def write_sharded(
    split: str,
    files: List[str],
    outdir: str,
    dataset_tag: str,
    vec: EmberV3Vectorizer,
    label_key: str,
    id_key: str,
    max_samples_per_file: int = 0,
    debug_first_error: bool = True,
):
    split_dir = os.path.join(outdir, split.lower())
    ensure_dir(split_dir)

    manifest = {
        "dataset_tag": dataset_tag,
        "split": split,
        "feature_dim": FEATURE_DIM,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "items": []
    }
    manifest_state = {"printed_first_error": False}

    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        prefix = f"{dataset_tag}__{base}"

        # smoke test BEFORE allocating huge output
        try:
            smoke_test_jsonl(fp, vec, label_key, id_key)
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Smoke test failed for {fp}:\n{e}\n{tb}")

        n_lines = wc_l(fp)
        n = min(n_lines, max_samples_per_file) if max_samples_per_file > 0 else n_lines

        X_path = os.path.join(split_dir, f"{prefix}__X_tab_emberv3.npy")
        y_path = os.path.join(split_dir, f"{prefix}__y.npy")
        s_path = os.path.join(split_dir, f"{prefix}__sha256.npy")
        v_path = os.path.join(split_dir, f"{prefix}__valid.npy")
        m_path = os.path.join(split_dir, f"{prefix}__meta.json")

        X = open_memmap(X_path, mode="w+", dtype=np.float32, shape=(n, FEATURE_DIM))
        y = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(n,))
        sha = open_memmap(s_path, mode="w+", dtype="S64", shape=(n,))
        valid = open_memmap(v_path, mode="w+", dtype=np.uint8, shape=(n,))

        errors = 0
        written = 0
        pos = 0
        n_valid = 0
        t0 = time.time()

        with open(fp, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if written >= n:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    lab = int(obj.get(label_key, 0))
                    sid_str = str(obj.get(id_key, ""))
                    sid = sid_str.encode("utf-8")[:64]

                    X[written, :] = vec.transform(obj)
                    y[written] = lab
                    sha[written] = sid
                    valid[written] = 1
                    n_valid += 1
                    pos += lab

                except Exception as e:
                    errors += 1
                    X[written, :] = 0
                    y[written] = 0
                    sha[written] = b""
                    valid[written] = 0

                    if debug_first_error:
                        _print_first_error_once(
                            manifest_state,
                            msg=(
                                f"File: {fp}\n"
                                f"Line number: {line_no}\n"
                                f"Exception: {repr(e)}\n"
                                f"sha256 (if present): {obj.get(id_key) if 'obj' in locals() else 'N/A'}\n"
                                f"Traceback:\n{traceback.format_exc()}"
                            )
                        )

                written += 1

        pos_rate = (pos / n_valid) if n_valid else None
        dt = time.time() - t0

        meta = {
            "dataset_tag": dataset_tag,
            "split": split,
            "source_jsonl": fp,
            "n_lines_in_file": n_lines,
            "n_rows_written": written,
            "n_valid": n_valid,
            "pos_rate_valid": pos_rate,
            "errors": errors,
            "feature_dim": FEATURE_DIM,
            "label_key": label_key,
            "id_key": id_key,
            "strict_exports_bug": vec.strict_exports_bug,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "cache_files": {"X": X_path, "y": y_path, "sha256": s_path, "valid": v_path}
        }
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[{split}] {base}: wrote={written} valid={n_valid} errors={errors} pos_rate={pos_rate} time={dt:.1f}s")

        manifest["items"].append({
            "source_jsonl": fp,
            "base": base,
            "meta": m_path,
            "X": X_path, "y": y_path, "sha256": s_path, "valid": v_path,
            "n_rows_written": written,
            "n_valid": n_valid,
            "pos_rate_valid": pos_rate,
            "errors": errors
        })

    manifest_path = os.path.join(outdir, f"manifest_{split.lower()}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest:", manifest_path)


def write_single(
    split: str,
    files: List[str],
    outdir: str,
    dataset_tag: str,
    vec: EmberV3Vectorizer,
    label_key: str,
    id_key: str,
    max_samples_total: int = 0,
    debug_first_error: bool = True,
):
    ensure_dir(outdir)
    state = {"printed_first_error": False}

    # smoke test first file
    smoke_test_jsonl(files[0], vec, label_key, id_key)

    counts = []
    total = 0
    for fp in files:
        n = wc_l(fp)
        counts.append((fp, n))
        total += n

    if max_samples_total and max_samples_total < total:
        total = max_samples_total

    X_path = os.path.join(outdir, f"{dataset_tag}_{split}_X_tab_emberv3.npy")
    y_path = os.path.join(outdir, f"{dataset_tag}_{split}_y.npy")
    s_path = os.path.join(outdir, f"{dataset_tag}_{split}_sha256.npy")
    v_path = os.path.join(outdir, f"{dataset_tag}_{split}_valid.npy")
    idx_path = os.path.join(outdir, f"{dataset_tag}_{split}_index.json")
    meta_path = os.path.join(outdir, f"{dataset_tag}_{split}_meta.json")

    X = open_memmap(X_path, mode="w+", dtype=np.float32, shape=(total, FEATURE_DIM))
    y = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(total,))
    sha = open_memmap(s_path, mode="w+", dtype="S64", shape=(total,))
    valid = open_memmap(v_path, mode="w+", dtype=np.uint8, shape=(total,))

    index = []
    row = 0
    errors = 0
    pos = 0
    n_valid = 0
    t0 = time.time()

    for fp, _n_lines in counts:
        start = row
        with open(fp, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if row >= total:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lab = int(obj.get(label_key, 0))
                    sid = str(obj.get(id_key, "")).encode("utf-8")[:64]
                    X[row, :] = vec.transform(obj)
                    y[row] = lab
                    sha[row] = sid
                    valid[row] = 1
                    n_valid += 1
                    pos += lab
                except Exception as e:
                    errors += 1
                    X[row, :] = 0
                    y[row] = 0
                    sha[row] = b""
                    valid[row] = 0

                    if debug_first_error:
                        _print_first_error_once(
                            state,
                            msg=(
                                f"File: {fp}\n"
                                f"Line number: {line_no}\n"
                                f"Exception: {repr(e)}\n"
                                f"sha256 (if present): {obj.get(id_key) if 'obj' in locals() else 'N/A'}\n"
                                f"Traceback:\n{traceback.format_exc()}"
                            )
                        )

                row += 1

        end = row
        index.append({"source_jsonl": fp, "start_row": start, "end_row": end})
        if row >= total:
            break

    pos_rate = (pos / n_valid) if n_valid else None
    dt = time.time() - t0

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"dataset_tag": dataset_tag, "split": split, "items": index}, f, indent=2)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_tag": dataset_tag,
            "split": split,
            "feature_dim": FEATURE_DIM,
            "label_key": label_key,
            "id_key": id_key,
            "n_rows_written": row,
            "n_valid": n_valid,
            "pos_rate_valid": pos_rate,
            "errors": errors,
            "strict_exports_bug": vec.strict_exports_bug,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "files": {"X": X_path, "y": y_path, "sha256": s_path, "valid": v_path, "index": idx_path}
        }, f, indent=2)

    print(f"[{split} SINGLE] wrote={row}/{total} valid={n_valid} errors={errors} pos_rate={pos_rate} time={dt:.1f}s")
    print("  X:", X_path)
    print("  index:", idx_path)
    print("  meta:", meta_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train file or folder")
    ap.add_argument("--test", required=True, help="Test file or folder")
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--warnings_file", required=True, help="Path to THREMBER pefile_warnings.txt (87 lines)")
    ap.add_argument("--train_glob", default="*_train.jsonl")
    ap.add_argument("--test_glob", default="*_test.jsonl")
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--mode", choices=["sharded", "single"], default="sharded")
    ap.add_argument("--max_files_train", type=int, default=0)
    ap.add_argument("--max_files_test", type=int, default=0)
    ap.add_argument("--max_samples_per_file", type=int, default=0, help="(sharded) limit per jsonl")
    ap.add_argument("--max_samples_total", type=int, default=0, help="(single) limit total rows")
    ap.add_argument("--fix_exports_count", action="store_true")

    ap.add_argument("--no_debug_first_error", action="store_true", help="Disable printing first traceback")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    vec = EmberV3Vectorizer(
        strict_exports_bug=(not args.fix_exports_count),
        warnings_file=args.warnings_file
    )

    train_files = list_jsonl(args.train, args.train_glob, recursive=args.recursive)
    test_files = list_jsonl(args.test, args.test_glob, recursive=args.recursive)

    if args.max_files_train:
        train_files = train_files[:args.max_files_train]
    if args.max_files_test:
        test_files = test_files[:args.max_files_test]

    print("Train files:", len(train_files), "| Test files:", len(test_files))
    print("Mode:", args.mode)
    print("Feature dim:", FEATURE_DIM)
    print("strict_exports_bug:", vec.strict_exports_bug)

    debug_first_error = (not args.no_debug_first_error)

    if args.mode == "sharded":
        write_sharded("TRAIN", train_files, args.outdir, args.dataset_tag, vec, "label", "sha256",
                      max_samples_per_file=args.max_samples_per_file,
                      debug_first_error=debug_first_error)
        write_sharded("TEST", test_files, args.outdir, args.dataset_tag, vec, "label", "sha256",
                      max_samples_per_file=args.max_samples_per_file,
                      debug_first_error=debug_first_error)
    else:
        write_single("TRAIN", train_files, args.outdir, args.dataset_tag, vec, "label", "sha256",
                    max_samples_total=args.max_samples_total,
                    debug_first_error=debug_first_error)
        write_single("TEST", test_files, args.outdir, args.dataset_tag, vec, "label", "sha256",
                    max_samples_total=args.max_samples_total,
                    debug_first_error=debug_first_error)


if __name__ == "__main__":
    main()
