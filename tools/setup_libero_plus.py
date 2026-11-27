"""Install the pinned LIBERO Plus assets and write an isolated simulator config."""
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from huggingface_hub import hf_hub_download

CODE_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"


def install(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "LIBERO-plus"
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/sylvestf/LIBERO-plus.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", CODE_REVISION], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != CODE_REVISION:
        raise RuntimeError(f"Check out the pinned LIBERO Plus revision: {CODE_REVISION}")
    base = repo / "libero/libero"
    config_dir = root / "plus_config"
    config_dir.mkdir(exist_ok=True)
    # The manifest and text cache are generated separately from the asset install.
    (root / "data").mkdir(exist_ok=True)
    marker = root / "plus_assets_revision.txt"
    if not marker.exists():
        archive = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset",
                                  revision=ASSET_REVISION, local_dir=root / "downloads")
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                parts = PurePosixPath(member.filename).parts
                if "assets" not in parts or member.is_dir():
                    continue
                target = (base / Path(*parts[parts.index("assets"):])).resolve()
                if not target.is_relative_to(base / "assets"):
                    raise ValueError(f"Invalid asset path: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        if not (base / "assets/scenes").is_dir():
            raise RuntimeError("The archive did not contain assets/scenes")
        marker.write_text(ASSET_REVISION + "\n")
    if marker.read_text().strip() != ASSET_REVISION or not (base / "assets/scenes").is_dir():
        raise RuntimeError("Missing or mismatched LIBERO Plus assets")
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"), "assets": str(base / "assets"),
        "datasets": str(root / "data/raw"),
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    install(parser.parse_args().root)
