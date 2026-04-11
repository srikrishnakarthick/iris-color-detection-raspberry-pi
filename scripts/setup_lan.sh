#!/bin/bash

# ==========================================
# Raspberry Pi ↔ Windows LAN Setup Script
# ==========================================

# ----------- USER CONFIG -------------------
CONNECTION_NAME="YOUR_CONNECTION_NAME"     # e.g., "Wired connection 1"
PI_IP="YOUR_PI_STATIC_IP/24"              # e.g., 192.168.x.x/24
GATEWAY_IP="YOUR_GATEWAY_IP"              # e.g., 192.168.x.1
DNS_IP="YOUR_DNS_IP"                      # usually same as gateway

WINDOWS_SHARE="//YOUR_WINDOWS_IP/YOUR_SHARE_NAME"
MOUNT_POINT="/mnt/pi_to_win"

USERNAME="YOUR_WINDOWS_USERNAME"
PASSWORD="YOUR_WINDOWS_PASSWORD"
# ------------------------------------------

echo "Setting static IP on Raspberry Pi..."

nmcli connection modify "$CONNECTION_NAME" \
    ipv4.method manual \
    ipv4.addresses "$PI_IP" \
    ipv4.gateway "$GATEWAY_IP" \
    ipv4.dns "$DNS_IP"

nmcli connection up "$CONNECTION_NAME"

echo "Verifying network configuration..."
ip addr show eth0

echo "Testing connection to Windows machine..."
ping -c 4 "$GATEWAY_IP"

echo "Installing required packages..."
sudo apt update
sudo apt install -y cifs-utils

echo "Creating mount directory..."
sudo mkdir -p "$MOUNT_POINT"

echo "Creating credentials file..."
sudo bash -c "cat > /root/.smbcredentials <<EOF
username=$USERNAME
password=$PASSWORD
EOF"

sudo chmod 600 /root/.smbcredentials

echo "Mounting Windows shared folder..."
sudo mount -t cifs "$WINDOWS_SHARE" "$MOUNT_POINT" \
    -o credentials=/root/.smbcredentials,uid=1000,gid=1000,iocharset=utf8,vers=3.0

echo "Testing write access..."
touch "$MOUNT_POINT/test.txt"

echo "Setting up auto-mount on reboot..."
sudo bash -c "echo '$WINDOWS_SHARE $MOUNT_POINT cifs credentials=/root/.smbcredentials,uid=1000,gid=1000,iocharset=utf8,vers=3.0,_netdev 0 0' >> /etc/fstab"

sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo "Remounting all filesystems..."
sudo mount -a

echo "Setup complete! Rebooting..."
sudo reboot

