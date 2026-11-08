"""Copy the project into an ERAV5 checkout's session13/ folder (no data, no venv),
fixing the Colab clone URL in notebooks that were executed before it was set."""
import json, pathlib, shutil, sys
src = pathlib.Path(__file__).resolve().parents[1]
dst = pathlib.Path(sys.argv[1]) / "session13"
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)
files = [*src.glob("*.ipynb"), *src.glob("*.py"), src / "README.md", src / "run_all.sh", src / ".gitignore"]
for f in files:
    shutil.copy2(f, dst / f.name)
for d in ["tools", "results", "figures"]:
    if (src / d).exists():
        shutil.copytree(src / d, dst / d, ignore=shutil.ignore_patterns("__pycache__"))
old = ["    !git clone -q https://github.com/REPLACE_ME/reversible-llm.git /content/reversible-llm\n",
       "    %cd /content/reversible-llm\n"]
new = ["    !git clone -q -b session13 https://github.com/chapanduk/ERAV5.git /content/ERAV5\n",
       "    %cd /content/ERAV5/session13\n"]
for nb in dst.glob("*.ipynb"):
    j = json.load(open(nb))
    for c in j["cells"]:
        c["source"] = [new[old.index(l)] if l in old else l for l in c["source"]]
    json.dump(j, open(nb, "w"), indent=1)
print("synced", len(list(dst.rglob("*"))), "files to", dst)
