#!/usr/bin/env bash

set -euo pipefail

STEP_CA_VERSION="0.30.2"
STEP_CLI_VERSION="0.30.6"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CERT_DIR="${SCRIPT_DIR}/certs"
STEPPATH="${STEPPATH:-${HOME}/.step}"
CA_NAME="${CA_NAME:-Taks-CA}"
CA_DNS="${CA_DNS:-router.taks.local}"
CA_PORT="${CA_PORT:-9000}"
PROVISIONER="${PROVISIONER:-admin}"
PASSWORD_FILE="${STEPPATH}/.password"
SERVER_SAN="${SERVER_SAN:-router.taks.local,localhost}"
CLIENT_NAME="${CLIENT_NAME:-robot}"
SERVER_DAYS="${SERVER_DAYS:-90}"
CLIENT_DAYS="${CLIENT_DAYS:-365}"
ARCH="$(uname -m)"

case "$ARCH" in
    x86_64|amd64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) echo "✗ 不支持的架构: $ARCH"; exit 1 ;;
esac

log() { echo "✓ $*"; }
err() { echo "✗ $*" >&2; exit 1; }

install_step() {
    if command -v step &>/dev/null && command -v step-ca &>/dev/null; then
        log "step / step-ca 已安装"
        return
    fi
    local tmpdir bindir
    tmpdir="$(mktemp -d)"
    trap "rm -rf $tmpdir" RETURN
    if [[ -w /usr/local/bin ]]; then
        bindir="/usr/local/bin"
    else
        bindir="${HOME}/.local/bin"
        mkdir -p "$bindir"
    fi
    log "下载 step-ca v${STEP_CA_VERSION} (${ARCH})..."
    curl -fsSL -o "${tmpdir}/step-ca.tar.gz" \
        "https://github.com/smallstep/certificates/releases/download/v${STEP_CA_VERSION}/step-ca_linux_${STEP_CA_VERSION}_${ARCH}.tar.gz"
    tar -xzf "${tmpdir}/step-ca.tar.gz" -C "${tmpdir}"
    if [[ -f "${tmpdir}/step-ca" ]]; then
        install -m 0755 "${tmpdir}/step-ca" "${bindir}/step-ca"
    else
        install -m 0755 "${tmpdir}/step-ca_${STEP_CA_VERSION}/bin/step-ca" "${bindir}/step-ca"
    fi
    log "下载 step CLI v${STEP_CLI_VERSION} (${ARCH})..."
    curl -fsSL -o "${tmpdir}/step.tar.gz" \
        "https://dl.smallstep.com/gh-release/cli/gh-release-header/v${STEP_CLI_VERSION}/step_linux_${STEP_CLI_VERSION}_${ARCH}.tar.gz"
    tar -xzf "${tmpdir}/step.tar.gz" -C "${tmpdir}"
    install -m 0755 "${tmpdir}/step_${STEP_CLI_VERSION}/bin/step" "${bindir}/step"
    log "step v${STEP_CLI_VERSION} / step-ca v${STEP_CA_VERSION} 安装到 ${bindir}"
}

ensure_password() {
    mkdir -p "$STEPPATH"
    if [[ ! -f "$PASSWORD_FILE" ]]; then
        openssl rand -base64 32 > "$PASSWORD_FILE"
        chmod 600 "$PASSWORD_FILE"
        log "已生成 CA 密码文件: $PASSWORD_FILE"
    fi
}

init_ca() {
    if [[ -f "${STEPPATH}/config/ca.json" ]]; then
        log "step-ca 已初始化"
        return
    fi
    ensure_password
    rm -rf "${STEPPATH}/config" "${STEPPATH}/certs" "${STEPPATH}/secrets" "${STEPPATH}/db"
    step ca init \
        --name "$CA_NAME" \
        --dns "$CA_DNS" \
        --dns "127.0.0.1" \
        --dns "localhost" \
        --address ":${CA_PORT}" \
        --provisioner "$PROVISIONER" \
        --password-file "$PASSWORD_FILE" \
        --deployment-type standalone
    local ca_json="${STEPPATH}/config/ca.json"
    python3 -c "
import json
with open('${ca_json}') as f:
    cfg = json.load(f)
auth = cfg.setdefault('authority', {})
claims = {
    'minTLSCertDuration': '5m',
    'maxTLSCertDuration': '${CLIENT_DAYS}h',
    'defaultTLSCertDuration': '${SERVER_DAYS}h',
    'maxDuration': '${CLIENT_DAYS}h',
    'defaultDuration': '${SERVER_DAYS}h',
}
auth['claims'] = claims
for p in auth.get('provisioners', []):
    p['claims'] = claims
with open('${ca_json}', 'w') as f:
    json.dump(cfg, f, indent='\t')
"
    log "step-ca 初始化完成 (maxTLSCertDuration=${CLIENT_DAYS}d)"
}

start_step_ca() {
    if pgrep -x step-ca &>/dev/null; then
        log "step-ca 已在运行"
        return
    fi
    local step_ca_bin
    step_ca_bin="$(command -v step-ca)"
    if sudo -n true 2>/dev/null && systemctl is-system-running &>/dev/null; then
        local svc="/etc/systemd/system/step-ca.service"
        sudo tee "$svc" >/dev/null <<EOF
[Unit]
Description=step-ca Taks CA
After=network.target

[Service]
Type=simple
ExecStart=${step_ca_bin} ${STEPPATH}/config/ca.json --password-file ${PASSWORD_FILE}
Environment=STEPPATH=${STEPPATH}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable --now step-ca
        log "step-ca systemd 服务已启动"
    else
        nohup step-ca "${STEPPATH}/config/ca.json" \
            --password-file "$PASSWORD_FILE" \
            >/tmp/step-ca.log 2>&1 &
        log "step-ca 后台进程已启动 (PID=$!)"
    fi
    sleep 2
}

get_ca_fingerprint() {
    step certificate fingerprint "${STEPPATH}/certs/root_ca.crt"
}

issue_cert() {
    local subject="$1" cert="$2" key="$3" sans="$4" days="$5"
    local san_args=()
    IFS=',' read -ra san_list <<< "$sans"
    for s in "${san_list[@]}"; do
        san_args+=("--san=$s")
    done
    step ca certificate "$subject" "$cert" "$key" \
        --kty EC --curve P-256 \
        --not-after "${days}h" \
        --provisioner "$PROVISIONER" \
        --provisioner-password-file "$PASSWORD_FILE" \
        --ca-url "https://127.0.0.1:${CA_PORT}" \
        --root "${STEPPATH}/certs/root_ca.crt" \
        --force \
        "${san_args[@]}"
    chmod 600 "$key"
    log "已签发: $cert (SAN: $sans, ECDSA P-256, ${days}h)"
}

export_ca() {
    cp "${STEPPATH}/certs/root_ca.crt" "${CERT_DIR}/ca-cert.pem"
    chmod 644 "${CERT_DIR}/ca-cert.pem"
    log "已导出 CA 证书: ${CERT_DIR}/ca-cert.pem"
}

setup_autorenew() {
    local renew_script="${CERT_DIR}/renew_certs.sh"
    cat > "$renew_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export STEPPATH="${STEPPATH}"
step ca renew --force "${CERT_DIR}/server-cert.pem" "${CERT_DIR}/server-key.pem" \
    --ca-url "https://127.0.0.1:${CA_PORT}" \
    --root "${STEPPATH}/certs/root_ca.crt"
step ca renew --force "${CERT_DIR}/client-cert.pem" "${CERT_DIR}/client-key.pem" \
    --ca-url "https://127.0.0.1:${CA_PORT}" \
    --root "${STEPPATH}/certs/root_ca.crt"
EOF
    chmod 700 "$renew_script"

    if sudo -n true 2>/dev/null && systemctl is-system-running &>/dev/null; then
        local timer="/etc/systemd/system/taks-cert-renew.timer"
        local svc="/etc/systemd/system/taks-cert-renew.service"
        sudo tee "$svc" >/dev/null <<EOF
[Unit]
Description=Taks 证书自动续期

[Service]
Type=oneshot
ExecStart=${renew_script}
EOF
        sudo tee "$timer" >/dev/null <<EOF
[Unit]
Description=Taks 证书自动续期定时器

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable --now taks-cert-renew.timer
        log "systemd 自动续期定时器已启用 (每天 03:00)"
    else
        local existing=""
        existing="$(crontab -l 2>/dev/null || true)"
        existing="$(echo "$existing" | grep -v "$renew_script" || true)"
        printf '%s\n0 3 * * * %s\n' "$existing" "$renew_script" | crontab -
        log "cron 自动续期已配置 (每天 03:00)，手动续期: ${renew_script}"
    fi
}

cmd_all() {
    install_step
    init_ca
    start_step_ca
    mkdir -p "$CERT_DIR"
    issue_cert "$CA_DNS" \
        "${CERT_DIR}/server-cert.pem" \
        "${CERT_DIR}/server-key.pem" \
        "$SERVER_SAN" "$SERVER_DAYS"
    issue_cert "$CLIENT_NAME" \
        "${CERT_DIR}/client-cert.pem" \
        "${CERT_DIR}/client-key.pem" \
        "$CLIENT_NAME.taks.local" "$CLIENT_DAYS"
    export_ca
    setup_autorenew
    echo ""
    log "全部完成，证书目录: $CERT_DIR"
    echo "  - CA 证书:     ${CERT_DIR}/ca-cert.pem"
    echo "  - Server 证书: ${CERT_DIR}/server-cert.pem"
    echo "  - Server 私钥: ${CERT_DIR}/server-key.pem"
    echo "  - Client 证书: ${CERT_DIR}/client-cert.pem"
    echo "  - Client 私钥: ${CERT_DIR}/client-key.pem"
}

cmd_init() {
    install_step
    init_ca
    start_step_ca
    log "CA 初始化完成"
}

cmd_issue() {
    mkdir -p "$CERT_DIR"
    issue_cert "$CA_DNS" \
        "${CERT_DIR}/server-cert.pem" \
        "${CERT_DIR}/server-key.pem" \
        "$SERVER_SAN" "$SERVER_DAYS"
    issue_cert "$CLIENT_NAME" \
        "${CERT_DIR}/client-cert.pem" \
        "${CERT_DIR}/client-key.pem" \
        "$CLIENT_NAME.taks.local" "$CLIENT_DAYS"
    export_ca
    log "证书签发完成"
}

cmd_renew() {
    local renew_script="${CERT_DIR}/renew_certs.sh"
    if [[ -x "$renew_script" ]]; then
        "$renew_script"
        log "证书续期完成"
    else
        err "续期脚本不存在，请先运行 $0 all 或 $0 issue"
    fi
}

usage() {
    cat <<EOF
Taks mTLS 证书管理 (step-ca v${STEP_CA_VERSION})

用法: $0 <子命令>
  all     安装 step-ca + 初始化 CA + 签发证书 + 配置自动续期
  init    仅初始化 CA 并启动 step-ca
  issue   仅签发 server/client 证书
  renew   手动续期证书

环境变量:
  CA_DNS        CA 域名 (默认: router.taks.local)
  CA_PORT       CA 端口 (默认: 9000)
  SERVER_SAN    Server SAN (默认: router.taks.local,localhost)
  CLIENT_NAME   Client 名称 (默认: robot)
  SERVER_DAYS   Server 有效期小时 (默认: 90)
  CLIENT_DAYS   Client 有效期小时 (默认: 365)
  STEPPATH      step 配置目录 (默认: ~/.step)
EOF
}

case "${1:-all}" in
    all)    cmd_all ;;
    init)   cmd_init ;;
    issue)  cmd_issue ;;
    renew)  cmd_renew ;;
    -h|--help) usage ;;
    *)      usage; exit 1 ;;
esac
