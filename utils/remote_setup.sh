#!/usr/bin/env bash
# Push this repo to remote Azure DGX A100 boxes and build the env on each.
#   utils/remote_setup.sh <host> [<host> ...]      # host = [user@]hostname
# Needs key-based SSH from this box to each remote. Safe to re-run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "$#" -gt 0 ] || { echo "usage: $0 <host> [<host> ...]" >&2; exit 2; }

SSH="ssh -o StrictHostKeyChecking=accept-new"
rc=0
for host in "$@"; do
    echo "==== $host ===="
    # 同步也遵守公开文件边界，避免发送本地笔记、凭据、模型和已安装环境。
    if rsync -az --delete -e "$SSH" \
        --filter=':- .gitignore' \
        --exclude '.git/' --exclude '/data/' --exclude '/metrics/' --exclude '/logs/' \
        --exclude '/results/' --exclude '__pycache__/' --exclude '*.pyc' \
        "$REPO_ROOT/" "$host:xixi-server/" \
        && $SSH "$host" 'cd xixi-server && bash utils/setup_repo.sh'; then
        echo "  OK: $host ready"
    else
        echo "  FAIL: $host" >&2; rc=1
    fi
done
exit $rc
