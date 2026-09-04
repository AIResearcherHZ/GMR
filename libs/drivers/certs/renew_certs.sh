#!/usr/bin/env bash
set -euo pipefail
CERT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CERT_DIR"

CFG="$(mktemp)"
trap 'rm -f "$CFG" /tmp/taks_server.csr /tmp/taks_client.csr' EXIT
cat > "$CFG" <<'EOF'
[req]
distinguished_name = dn
x509_extensions = v3_ca
prompt = no
[dn]
O = Haozheng Xie
CN = Haozheng Xie Root CA
[v3_ca]
basicConstraints = critical,CA:TRUE,pathlen:1
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
[server_ext]
basicConstraints = CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:router.taks.local,DNS:localhost,IP:127.0.0.1,IP:0.0.0.0
[client_ext]
basicConstraints = CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = clientAuth
subjectAltName = DNS:robot
EOF

openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout ca-key.pem -out ca-cert.pem -days 36500 -nodes -config "$CFG"

openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout server-key.pem -out /tmp/taks_server.csr -nodes \
  -subj "/CN=router.taks.local"
openssl x509 -req -in /tmp/taks_server.csr -CA ca-cert.pem -CAkey ca-key.pem \
  -CAcreateserial -out server-cert.pem -days 36500 \
  -extfile "$CFG" -extensions server_ext

openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout client-key.pem -out /tmp/taks_client.csr -nodes \
  -subj "/CN=robot"
openssl x509 -req -in /tmp/taks_client.csr -CA ca-cert.pem -CAkey ca-key.pem \
  -CAcreateserial -out client-cert.pem -days 36500 \
  -extfile "$CFG" -extensions client_ext

chmod 600 ca-key.pem server-key.pem client-key.pem
rm -f ca-cert.srl
openssl verify -CAfile ca-cert.pem server-cert.pem client-cert.pem
openssl x509 -in server-cert.pem -noout -subject -dates