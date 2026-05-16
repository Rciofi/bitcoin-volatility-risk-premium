"""
Sync script: local project <-> Overleaf "Dissertação Final 10122025"

Usage (from project root):
  python scripts/sync_overleaf.py pull       -- baixa .tex/.bib do Overleaf para local
  python scripts/sync_overleaf.py push_figs  -- envia figuras .png de figs/ para o Overleaf
  python scripts/sync_overleaf.py status     -- mostra diferenças sem alterar nada

NOTA: O comando 'push' (envio de .tex/.bib) não é suportado via REST API do Overleaf
cloud. Arquivos .tex são tratados como "documentos" internamente e só podem ser
atualizados via socket.io ou pelo Git do Overleaf. Use o Git sync do Overleaf para
sincronizar arquivos .tex:
  git remote add overleaf https://git.overleaf.com/<PROJECT_ID>
  git push overleaf main
"""

import io
import json
import pickle
import sys
import uuid
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ID = "69248f9218f9578f0df9ad94"
BASE_URL = "https://www.overleaf.com"
OLAUTH_PATH = Path(__file__).resolve().parents[1] / ".olauth"

SYNC_EXTENSIONS = {".tex", ".bib"}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def load_auth():
    with open(OLAUTH_PATH, "rb") as f:
        store = pickle.load(f)
    return store["cookie"], store["csrf"]


def get_project_info(cookie):
    """
    Retorna (csrf, tree) parseando a página do projeto.
    Overleaf injeta o CSRF em <meta name="ol-csrfToken"> e a árvore de pastas
    em <meta name="ol-rootFolder"> — não precisa de endpoint REST.
    """
    r = requests.get(f"{BASE_URL}/project/{PROJECT_ID}", cookies=cookie)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")

    csrf = None
    csrf_meta = soup.find("meta", {"name": "ol-csrfToken"})
    if csrf_meta:
        csrf = csrf_meta.get("content")

    tree = None
    tree_meta = soup.find("meta", {"name": "ol-rootFolder"})
    if tree_meta:
        content = tree_meta.get("content", "")
        if content:
            try:
                folders = json.loads(content)
                if folders:
                    tree = folders[0]
            except (json.JSONDecodeError, IndexError):
                pass

    return csrf, tree


# Mantido por compatibilidade, mas agora usa get_project_info
def get_csrf(cookie):
    csrf, _ = get_project_info(cookie)
    return csrf


def get_project_tree(cookie):
    _, tree = get_project_info(cookie)
    return tree


# ---------------------------------------------------------------------------
# Pull: Overleaf -> local
# ---------------------------------------------------------------------------

def pull():
    cookie, _ = load_auth()
    print("Baixando projeto do Overleaf...")

    r = requests.get(
        f"{BASE_URL}/project/{PROJECT_ID}/download/zip",
        cookies=cookie,
        stream=True,
    )
    r.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    updated, skipped = [], []

    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if Path(name).suffix not in SYNC_EXTENSIONS:
            continue

        remote_content = zf.read(name)
        local_path = Path(name)

        if local_path.exists() and local_path.read_bytes() == remote_content:
            skipped.append(name)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(remote_content)
        updated.append(name)

    print(f"\nAtualizados ({len(updated)}):")
    for f in updated:
        print(f"  + {f}")
    print(f"\nJá em dia ({len(skipped)}):")
    for f in skipped:
        print(f"  = {f}")
    print("\nPull concluído.")


# ---------------------------------------------------------------------------
# Utilitários de pasta
# ---------------------------------------------------------------------------

def _create_folder(cookie, csrf, parent_id, name):
    """Cria uma pasta no Overleaf e retorna seu _id."""
    r = requests.post(
        f"{BASE_URL}/project/{PROJECT_ID}/folder",
        cookies=cookie,
        headers={"X-Csrf-Token": csrf, "Accept": "application/json"},
        json={"name": name, "parent_folder_id": parent_id},
    )
    if r.ok:
        return r.json().get("_id")
    return None


def ensure_folder_path(cookie, csrf, tree, path_parts):
    """Garante que o caminho de pastas existe no Overleaf; retorna o folder_id folha."""
    current_node = tree
    current_id = tree["_id"]
    for part in path_parts:
        found = next((f for f in current_node.get("folders", []) if f["name"] == part), None)
        if found:
            current_id = found["_id"]
            current_node = found
        else:
            new_id = _create_folder(cookie, csrf, current_id, part)
            if not new_id:
                raise RuntimeError(f"Falha ao criar pasta '{part}' no Overleaf")
            print(f"  + pasta criada no Overleaf: {part}/")
            current_id = new_id
            current_node = {"_id": new_id, "folders": [], "fileRefs": [], "docs": []}
    return current_id


# ---------------------------------------------------------------------------
# Push figuras: local -> Overleaf (funciona para binários .png/.pdf)
# ---------------------------------------------------------------------------

def push_figs(figs=None):
    """
    Envia figuras PNG de figs/ (incluindo subpastas) para o Overleaf.
    Preserva a estrutura de subpastas (ex: figs/cap9/fig_cap9_01.png
    vai para a pasta cap9/ dentro de figs/ no Overleaf).

    Se figs for uma lista de caminhos relativos a partir de figs/, envia só essas.
    Ex: push_figs(['cap9/fig_cap9_01_rv30d_regimes.png'])
    """
    cookie, _ = load_auth()
    csrf, tree = get_project_info(cookie)

    if not csrf:
        raise RuntimeError("Não foi possível obter o CSRF token do Overleaf.")
    if not tree:
        raise RuntimeError("Não foi possível obter a estrutura do projeto Overleaf.")

    root = Path(__file__).resolve().parents[1]
    figs_dir = root / "figs"

    if figs:
        targets = [figs_dir / f for f in figs]
    else:
        targets = list(figs_dir.rglob("*.png"))

    print(f"Enviando {len(targets)} figura(s) para o Overleaf...\n")

    folder_id_cache: dict[tuple, str] = {(): tree["_id"]}

    ok_list, fail_list = [], []
    for fig_path in sorted(targets):
        if not fig_path.exists():
            print(f"  ! {fig_path.relative_to(root)} não encontrado localmente")
            continue

        rel = fig_path.relative_to(root)
        subfolder_parts = tuple(rel.parts[:-1])

        if subfolder_parts not in folder_id_cache:
            folder_id_cache[subfolder_parts] = ensure_folder_path(
                cookie, csrf, tree, list(subfolder_parts)
            )
        folder_id = folder_id_cache[subfolder_parts]

        params = {
            "folder_id": folder_id,
            "_csrf": csrf,
            "qquuid": str(uuid.uuid4()),
            "qqfilename": fig_path.name,
            "qqtotalfilesize": fig_path.stat().st_size,
        }
        with open(fig_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/project/{PROJECT_ID}/upload",
                cookies=cookie,
                headers={"X-Csrf-Token": csrf, "Accept": "application/json"},
                params=params,
                files={"qqfile": (fig_path.name, f, "image/png")},
            )

        rel_str = str(rel).replace("\\", "/")
        if resp.ok:
            ok_list.append(rel_str)
            print(f"  + {rel_str}")
        else:
            fail_list.append(rel_str)
            print(f"  x {rel_str} (status {resp.status_code}: {resp.text[:120]})")

    print(f"\nEnviados: {len(ok_list)} | Falhas: {len(fail_list)}")


# ---------------------------------------------------------------------------
# Status: mostra diferenças sem alterar
# ---------------------------------------------------------------------------

def collect_local_files():
    root = Path(__file__).resolve().parents[1]
    files = []
    for path in root.rglob("*"):
        if path.suffix in SYNC_EXTENSIONS and path.is_file():
            parts = path.relative_to(root).parts
            if parts[0] not in ("data", ".git", "scripts"):
                files.append(path)
    return files


def status():
    cookie, _ = load_auth()
    print("Comparando local vs Overleaf...\n")

    r = requests.get(
        f"{BASE_URL}/project/{PROJECT_ID}/download/zip",
        cookies=cookie,
        stream=True,
    )
    r.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    remote_files = {
        name: zf.read(name)
        for name in zf.namelist()
        if not name.endswith("/") and Path(name).suffix in SYNC_EXTENSIONS
    }

    root = Path(__file__).resolve().parents[1]
    local_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in collect_local_files()
    }

    only_remote = set(remote_files) - set(local_files)
    only_local = set(local_files) - set(remote_files)
    different = {
        k for k in remote_files if k in local_files and remote_files[k] != local_files[k]
    }
    in_sync = (set(remote_files) & set(local_files)) - different

    print(f"So no Overleaf ({len(only_remote)}):  (use pull para baixar)")
    for f in sorted(only_remote):
        print(f"  << {f}")

    print(f"\nSo local ({len(only_local)}):  (use push git para enviar)")
    for f in sorted(only_local):
        print(f"  >> {f}")

    print(f"\nDiferentes ({len(different)}):")
    for f in sorted(different):
        print(f"  != {f}")

    print(f"\nSincronizados ({len(in_sync)}):")
    for f in sorted(in_sync):
        print(f"  =  {f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pull":
        pull()
    elif cmd == "push_figs":
        figs_arg = sys.argv[2:] if len(sys.argv) > 2 else None
        push_figs(figs_arg)
    elif cmd == "status":
        status()
    elif cmd == "push":
        print(
            "AVISO: 'push' de arquivos .tex nao e suportado via REST API do Overleaf.\n"
            "Arquivos .tex sao 'documentos' no Overleaf e so podem ser atualizados\n"
            "via socket.io ou pelo Git sync do Overleaf.\n\n"
            "Para sincronizar .tex, configure o remote Git do Overleaf:\n"
            f"  git remote add overleaf https://git.overleaf.com/{PROJECT_ID}\n"
            "  git push overleaf main\n\n"
            "Para enviar figuras, use:\n"
            "  python scripts/sync_overleaf.py push_figs"
        )
        sys.exit(1)
    else:
        print(__doc__)
        sys.exit(1)
