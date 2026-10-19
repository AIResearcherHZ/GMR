#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

if [[ $EUID -ne 0 ]]; then
    exec sudo "$0" "$@"
fi

CAPS="cap_ipc_lock,cap_sys_nice+ep"
MANIFEST="/var/lib/taks_rt_caps.list"
REAL_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
[[ -n "$REAL_HOME" ]] || REAL_HOME="$HOME"

declare -a RUNAS=()
[[ -n "${SUDO_USER:-}" ]] && RUNAS=(sudo -u "$SUDO_USER" -H)

is_elf() {
    [[ "$(head -c4 "$1" 2>/dev/null | od -An -tx1 | tr -d ' \n')" == "7f454c46" ]]
}

emit_prefix() {
    local p
    for p in "$1"/bin/python "$1"/bin/python3 "$1"/bin/python3.*; do
        [[ -x "$p" ]] && printf '%s\n' "$p"
    done
}

discover() {
    local -a roots=(/usr/bin /usr/local/bin "$REAL_HOME/.local/bin")
    local base e d r p cand line pre uv_bin conda_bin
    for base in "$REAL_HOME/anaconda3" "$REAL_HOME/miniconda3" "$REAL_HOME/miniforge3" /opt/conda; do
        [[ -d "$base" ]] || continue
        roots+=("$base/bin")
        for e in "$base"/envs/*/bin; do roots+=("$e"); done
    done
    for d in "$REAL_HOME"/.conda/envs/*/bin "$REAL_HOME"/.virtualenvs/*/bin \
        "$REAL_HOME"/.local/share/uv/python/*/bin "$REAL_HOME"/.local/share/uv/tools/*/bin \
        "$REAL_HOME"/.pyenv/versions/*/bin; do
        roots+=("$d")
    done
    for r in "${roots[@]}"; do
        for p in "$r"/python "$r"/python3 "$r"/python3.*; do
            [[ -x "$p" ]] && printf '%s\n' "$p"
        done
    done

    conda_bin=""
    for cand in "$REAL_HOME"/anaconda3/bin/conda "$REAL_HOME"/miniconda3/bin/conda \
        "$REAL_HOME"/miniforge3/bin/conda /opt/conda/bin/conda; do
        [[ -x "$cand" ]] && { conda_bin="$cand"; break; }
    done
    if [[ -n "$conda_bin" ]]; then
        while IFS= read -r line; do
            [[ "$line" == \
            pre="${line##* }"
            [[ -d "$pre" ]] && emit_prefix "$pre"
        done < <("${RUNAS[@]}" "$conda_bin" env list 2>/dev/null || true)
    fi

    uv_bin=""
    for cand in "$REAL_HOME"/.local/bin/uv "$REAL_HOME"/.cargo/bin/uv \
        "$REAL_HOME"/anaconda3/bin/uv "$REAL_HOME"/anaconda3/envs/*/bin/uv \
        "$REAL_HOME"/miniconda3/bin/uv "$REAL_HOME"/miniconda3/envs/*/bin/uv \
        /usr/local/bin/uv /usr/bin/uv; do
        [[ -x "$cand" ]] && { uv_bin="$cand"; break; }
    done
    if [[ -n "$uv_bin" ]]; then
        "${RUNAS[@]}" "$uv_bin" python list --only-installed 2>/dev/null | awk 'NF>=2 {print $2}' || true
    fi
}

declare -a SOURCES=("$@")
mkdir -p "$(dirname "$MANIFEST")"
touch "$MANIFEST"

declare -A SEEN=()
count=0
while IFS= read -r py; do
    [[ -n "$py" && -x "$py" ]] || continue
    real="$(readlink -f "$py")"
    [[ -z "$real" || -n "${SEEN[$real]:-}" ]] && continue
    SEEN[$real]=1
    if ! is_elf "$real"; then
        echo "跳过(非 ELF): $py"
        continue
    fi
    setcap "$CAPS" "$real"
    echo "$real" >>"$MANIFEST"
    printf '✓ %s -> %s\n' "$py" "$(getcap "$real")"
    count=$((count + 1))
done < <(if [[ ${

sort -u "$MANIFEST" -o "$MANIFEST"
echo "已授权 $count 个解释器(CAP_IPC_LOCK + CAP_SYS_NICE),记录于 $MANIFEST。"
echo "此后直接运行这些解释器(无需 sudo)即自动获得内存锁定与实时调度。"
echo "一键撤销:bash $(cd "$(dirname "$0")" && pwd)/revoke_rt_caps.sh"
