"""Execute a notebook in place from this folder:  python _run_notebook.py NAME.ipynb [timeout_s]"""
import os, sys, nbformat
from nbclient import NotebookClient
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, sys.argv[1])
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 3600, kernel_name="python3",
               resources={"metadata": {"path": here}}).execute()
nbformat.write(nb, path)
print("executed", sys.argv[1])
