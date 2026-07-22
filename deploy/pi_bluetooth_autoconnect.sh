#!/usr/bin/env bash
# Make a Bluetooth speaker auto-reconnect on every boot on DietPi (root).
#
# Reboots wipe three things: the PulseAudio daemon (root), its bluetooth module,
# and the A2DP connection. This installs a systemd service that restores all
# three at boot so the speaker "just works" after a restart -- important for a
# live demo where you can't run 4 manual commands.
#
# Run ONCE on the Pi (as root):
#   BT_MAC=CF:63:A2:0F:54:81 bash deploy/pi_bluetooth_autoconnect.sh
#
# Then reboot to verify:  sudo reboot   (speaker should reconnect on its own)

set -euo pipefail

BT_MAC="${BT_MAC:?set BT_MAC=<speaker MAC>, e.g. CF:63:A2:0F:54:81}"
SINK="bluez_sink.$(echo "$BT_MAC" | tr ':' '_').a2dp_sink"

# --- the boot-time reconnect script ---
cat > /usr/local/bin/bt-speaker-connect.sh <<EOF
#!/usr/bin/env bash
# Restore PulseAudio + bluetooth module + connect the speaker. Idempotent.
set -x
MAC="$BT_MAC"
SINK="$SINK"

# 1) PulseAudio as root (survives no login session).
pulseaudio --check 2>/dev/null || pulseaudio --daemonize=yes --exit-idle-time=-1
sleep 2

# 2) Bluetooth audio module.
pactl load-module module-bluetooth-discover 2>/dev/null || true

# 3) Bring up the controller and connect, retrying (speaker may wake slowly).
bluetoothctl power on
for i in 1 2 3 4 5 6; do
    bluetoothctl connect "\$MAC" && break
    sleep 5
done

# 4) Make it the default sink and stop idle-suspend (cheap speakers drop on suspend).
sleep 2
pactl set-default-sink "\$SINK" 2>/dev/null || true
pactl unload-module module-suspend-on-idle 2>/dev/null || true
EOF
chmod +x /usr/local/bin/bt-speaker-connect.sh

# --- systemd service that runs it at boot ---
cat > /etc/systemd/system/bt-speaker.service <<EOF
[Unit]
Description=Auto-connect Bluetooth speaker + PulseAudio for the translator demo
After=bluetooth.service sound.target
Wants=bluetooth.service

[Service]
Type=oneshot
RemainAfterExit=yes
# Give bluetooth/audio a moment to come up before we connect.
ExecStartPre=/bin/sleep 8
ExecStart=/usr/local/bin/bt-speaker-connect.sh
# Retry the whole thing if the speaker was still waking up.
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable bt-speaker.service

cat <<EOF

[ok] Installed bt-speaker.service (auto-connects $BT_MAC on boot).
     Sink: $SINK

  Start it now:      sudo systemctl start bt-speaker.service
  Check status:      systemctl status bt-speaker.service
  View logs:         journalctl -u bt-speaker.service -b
  Test after reboot: sudo reboot   (speaker should reconnect by itself)

  If it fails to connect, the speaker was likely off/asleep -- power it on and:
      sudo systemctl restart bt-speaker.service
EOF
