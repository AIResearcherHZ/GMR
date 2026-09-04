#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    exec sudo "$0" "$@"
fi

MANIFEST="/var/lib/taks_rt_caps.list"
if [[ ! -s "$MANIFEST" ]]; then
    echo "无授权记录($MANIFEST 为空或不存在),没有需要撤销的。"
    exit 0
fi

count=0
while IFS= read -r real; do
    [[ -n "$real" && -e "$real" ]] || continue
    setcap -r "$real" 2>/dev/null || true
    printf '✗ 已撤销 %s (%s)\n' "$real" "$(getcap "$real" 2>/dev/null || echo none)"
    count=$((count + 1))
done <"$MANIFEST"

: >"$MANIFEST"
echo "已撤销 $count 个解释器的能力,并清空记录 $MANIFEST。"
