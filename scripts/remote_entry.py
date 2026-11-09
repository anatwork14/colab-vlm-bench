"""Run via 'colab exec -s SESSION -f scripts/remote_entry.py' after Drive mount."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

def main():
    archive=Path("/content/vlmbench_checkout.zip")
    camp=Path("/content/vlmbench_campaign.json")
    jobp=Path("/content/vlmbench_job.json")
    for p in (archive,camp,jobp):
        if not p.is_file(): raise FileNotFoundError(p)
    job=json.loads(jobp.read_text(encoding="utf8"))
    if job.get("owner_approved_compute") is not True:
        raise PermissionError("GPU execution requires owner_approved_compute=True")
    checkout=Path("/content/vlmbench_source")
    checkout.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p=Path(info.filename)
            if p.is_absolute() or ".." in p.parts: raise ValueError("Unsafe source archive")
        z.extractall(checkout)
    cfg=json.loads(camp.read_text(encoding="utf8"))
    if cfg.get("dry_run",True) or job["campaign_id"]!=cfg["campaign_id"]:
        raise RuntimeError("GPU job refused: config mismatch or dry run")
    if not Path("/content/drive").exists():
        raise RuntimeError("Drive unmounted: operator must run colab drivemount")
    subprocess.run([sys.executable,"-m","pip","install","-e",str(checkout)],check=True)
    os.environ["VLMBENCH_SOURCE_SHA"]=job["source_sha"]
    cmd=[sys.executable,"-m","vlmbench.cli","run-job","--config",str(camp),
         "--model",job["model"],"--mode",job["mode"],
         "--dataset",job["dataset"],"--max-cells",str(job["max_cells"]),"--approve-paid"]
    if job["quality"]: cmd.append("--quality")
    try:
        subprocess.run(cmd,check=True)
    finally:
        root=Path(cfg["storage"]["local_scratch_dir"])/cfg["campaign_id"]
        if root.is_dir():
            subprocess.run([sys.executable,"-m","vlmbench.cli","report",
                            "--config",str(camp),"--root",str(root)],check=False)
            exported=Path("/content/vlmbench_export_"+cfg["campaign_id"]+".zip")
            with zipfile.ZipFile(exported,"w",compression=zipfile.ZIP_DEFLATED) as z:
                for name in ("RUNBOOK.md","summary.json","environment.json","run_status.json"):
                    f=root/name
                    if f.is_file(): z.write(f,arcname=name)
            print("Exported",exported)

if __name__=="__main__": main()
