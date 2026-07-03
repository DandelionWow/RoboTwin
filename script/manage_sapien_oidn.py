#!/usr/bin/env python3
import argparse
import importlib.util
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


REQUIRED_LIBRARY_PATTERNS = (
    "libOpenImageDenoise.so*",
    "libOpenImageDenoise_core.so*",
    "libOpenImageDenoise_device_cuda.so*",
)
OIDN_DIR_NAME = "oidn_library"


class OidnLibraryError(RuntimeError):
    pass


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _version_key(path):
    nums = re.findall(r"\d+", path.name)
    return tuple(int(num) for num in nums)


def find_sapien_package_path():
    spec = importlib.util.find_spec("sapien")
    if spec is None:
        raise OidnLibraryError("Could not find the installed sapien package.")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    raise OidnLibraryError("Could not resolve the installed sapien package path.")


def get_oidn_library_path(sapien_package_path=None):
    package_path = Path(sapien_package_path).resolve() if sapien_package_path else find_sapien_package_path()
    return package_path, package_path / OIDN_DIR_NAME


def list_oidn_libraries(oidn_library_path):
    oidn_library_path = Path(oidn_library_path)
    return sorted(oidn_library_path.glob("libOpenImageDenoise*.so*"), key=lambda p: p.name)


def parse_versions(libraries):
    versions = set()
    for library in libraries:
        match = re.search(r"\.so\.([0-9][0-9.]+)", library.name)
        if match:
            versions.add(match.group(1).rstrip("."))
    return sorted(versions)


def validate_source_library(source):
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise OidnLibraryError(f"OIDN source directory does not exist: {source}")
    if not source.is_dir():
        raise OidnLibraryError(f"OIDN source path is not a directory: {source}")

    missing = [pattern for pattern in REQUIRED_LIBRARY_PATTERNS if not list(source.glob(pattern))]
    if missing:
        raise OidnLibraryError(
            "OIDN source directory is missing required libraries: "
            + ", ".join(missing)
        )
    return source


def _oidn_tricks_path(sapien_package_path):
    candidates = sorted(Path(sapien_package_path).rglob("_oidn_tricks.py"))
    return candidates[0] if candidates else None


def _fixed_library_names_from_oidn_tricks(sapien_package_path):
    tricks_path = _oidn_tricks_path(sapien_package_path)
    if tricks_path is None:
        return [], None
    text = tricks_path.read_text(encoding="utf-8", errors="replace")
    names = sorted(
        set(
            re.findall(
                r"['\"](libOpenImageDenoise(?:_[A-Za-z0-9]+)?\.so(?:\.[0-9][0-9.]*)?)['\"]",
                text,
            )
        )
    )
    return names, tricks_path


def _library_prefix(library_name):
    marker = ".so"
    idx = library_name.find(marker)
    if idx == -1:
        return library_name
    return library_name[: idx + len(marker)]


def _select_compatible_target(oidn_library_path, fixed_name):
    exact = oidn_library_path / fixed_name
    if exact.exists():
        return exact
    prefix = _library_prefix(fixed_name)
    candidates = [
        path
        for path in oidn_library_path.glob(prefix + "*")
        if path.name != fixed_name and path.is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_version_key, reverse=True)[0]


def ensure_compatible_symlinks(oidn_library_path, sapien_package_path, dry_run=False):
    oidn_library_path = Path(oidn_library_path)
    fixed_names, tricks_path = _fixed_library_names_from_oidn_tricks(sapien_package_path)
    if not fixed_names:
        if tricks_path is None:
            print("[RoboTwin] No _oidn_tricks.py found; no fixed OIDN filename symlinks needed.")
        else:
            print(f"[RoboTwin] No fixed OIDN filenames detected in {tricks_path}.")
        return

    print(f"[RoboTwin] Checking fixed OIDN filenames from {tricks_path}.")
    for fixed_name in fixed_names:
        link_path = oidn_library_path / fixed_name
        if link_path.exists():
            print(f"[RoboTwin] Compatible library exists: {link_path.name}")
            continue

        target = _select_compatible_target(oidn_library_path, fixed_name)
        if target is None:
            print(
                f"[RoboTwin] Warning: cannot create {fixed_name}; "
                f"no matching {_library_prefix(fixed_name)}* library found."
            )
            continue

        if dry_run:
            print(f"[RoboTwin] Dry run: would symlink {link_path.name} -> {target.name}")
        else:
            link_path.symlink_to(target.name)
            print(f"[RoboTwin] Created symlink {link_path.name} -> {target.name}")


def status():
    package_path, oidn_library_path = get_oidn_library_path()
    print(f"[RoboTwin] SAPIEN package path: {package_path}")
    print(f"[RoboTwin] SAPIEN oidn_library path: {oidn_library_path}")

    if not oidn_library_path.exists():
        raise OidnLibraryError(f"OIDN library directory does not exist: {oidn_library_path}")

    libraries = list_oidn_libraries(oidn_library_path)
    if libraries:
        print("[RoboTwin] OIDN libraries:")
        for library in libraries:
            print(f"  {library.name}")
    else:
        print("[RoboTwin] No libOpenImageDenoise*.so* files found.")

    versions = parse_versions(libraries)
    if versions:
        print(f"[RoboTwin] Detected OIDN version(s): {', '.join(versions)}")
    else:
        print("[RoboTwin] Could not parse OIDN versions from library filenames.")


def use_custom_oidn_library(source, dry_run=False):
    source = validate_source_library(source)
    package_path, oidn_library_path = get_oidn_library_path()
    parent = oidn_library_path.parent

    print(f"[RoboTwin] SAPIEN package path: {package_path}")
    print(f"[RoboTwin] Current oidn_library path: {oidn_library_path}")
    print(f"[RoboTwin] Custom OIDN source path: {source}")

    if not oidn_library_path.exists() or not oidn_library_path.is_dir():
        raise OidnLibraryError(f"Current oidn_library directory does not exist: {oidn_library_path}")

    backup_path = parent / f"{OIDN_DIR_NAME}.backup_{_timestamp()}"
    if dry_run:
        print(f"[RoboTwin] Dry run: would back up {oidn_library_path} to {backup_path}")
        print(f"[RoboTwin] Dry run: would copy {source} to {oidn_library_path}")
        ensure_compatible_symlinks(source, package_path, dry_run=True)
        return

    pending_path = parent / f"{OIDN_DIR_NAME}.pending_{_timestamp()}"
    if pending_path.exists():
        raise OidnLibraryError(f"Temporary path already exists: {pending_path}")

    print(f"[RoboTwin] Backing up existing OIDN library to {backup_path}")
    shutil.move(str(oidn_library_path), str(backup_path))
    try:
        shutil.copytree(source, pending_path, symlinks=True)
        ensure_compatible_symlinks(pending_path, package_path, dry_run=False)
        pending_path.rename(oidn_library_path)
    except Exception:
        if pending_path.exists():
            shutil.rmtree(pending_path)
        if not oidn_library_path.exists() and backup_path.exists():
            shutil.move(str(backup_path), str(oidn_library_path))
        raise

    print(f"[RoboTwin] Replaced OIDN library with: {source}")
    print(f"[RoboTwin] Backup saved at: {backup_path}")


def _latest_backup(parent):
    backups = sorted(parent.glob(f"{OIDN_DIR_NAME}.backup_*"), key=lambda p: p.name)
    if not backups:
        raise OidnLibraryError(f"No backups found in: {parent}")
    return backups[-1]


def restore_oidn_library(backup=None, latest=False):
    package_path, oidn_library_path = get_oidn_library_path()
    parent = oidn_library_path.parent
    backup_path = _latest_backup(parent) if latest else Path(backup).expanduser().resolve()

    if not backup_path.exists() or not backup_path.is_dir():
        raise OidnLibraryError(f"Backup directory does not exist: {backup_path}")

    current_backup = parent / f"{OIDN_DIR_NAME}.backup_before_restore_{_timestamp()}"
    print(f"[RoboTwin] SAPIEN package path: {package_path}")
    print(f"[RoboTwin] Restoring OIDN library from: {backup_path}")
    if oidn_library_path.exists():
        print(f"[RoboTwin] Backing up current OIDN library to: {current_backup}")
        shutil.move(str(oidn_library_path), str(current_backup))

    try:
        shutil.copytree(backup_path, oidn_library_path, symlinks=True)
    except Exception:
        if oidn_library_path.exists():
            shutil.rmtree(oidn_library_path)
        if current_backup.exists():
            shutil.move(str(current_backup), str(oidn_library_path))
        raise

    print(f"[RoboTwin] Restored OIDN library to: {oidn_library_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Manage SAPIEN bundled OIDN libraries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show current SAPIEN OIDN library information.")

    use_custom = subparsers.add_parser("use-custom", help="Replace SAPIEN OIDN libraries with a custom directory.")
    use_custom.add_argument("--source", required=True, help="Path to a replacement oidn_library directory.")
    use_custom.add_argument("--dry-run", action="store_true", help="Validate and show planned changes only.")

    restore = subparsers.add_parser("restore", help="Restore a SAPIEN OIDN library backup.")
    group = restore.add_mutually_exclusive_group(required=True)
    group.add_argument("--backup", help="Path to a backup directory.")
    group.add_argument("--latest", action="store_true", help="Restore the latest oidn_library.backup_* directory.")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            status()
        elif args.command == "use-custom":
            use_custom_oidn_library(args.source, dry_run=args.dry_run)
        elif args.command == "restore":
            restore_oidn_library(backup=args.backup, latest=args.latest)
    except OidnLibraryError as exc:
        print(f"[RoboTwin] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
