#!/bin/sh
# Usage (as root): perf_mode.sh on | off | status
# on     : max power mode + jetson_clocks + fan forced to maximum (via /tmp/fan_force_max, read by fan_daemon.sh). Use around benchmark runs only.
# off    : back to MODE_10W_DESKTOP, previous clocks, fan under temperature control again.
# status : print power mode, fan state and temperatures.
# Mode IDs are for Xavier NX (nvpmodel.conf): 5 = MODE_10W_DESKTOP, 8 = MODE_20W_6CORE.
STATE=/tmp/jetson_clocks_state.conf

fan_state() {
  for d in /sys/class/hwmon/hwmon*; do
    case "$(cat $d/name 2>/dev/null)" in
      pwmfan) pwm=$(cat $d/pwm1) ;;
      pwm_tach) rpm=$(cat $d/rpm 2>/dev/null) ;;
    esac
  done
  if [ -d /sys/devices/pwm-fan ]; then pwm=$(cat /sys/devices/pwm-fan/cur_pwm); fi
  echo "fan: pwm=$pwm rpm=${rpm:-n/a} forced_max=$([ -f /tmp/fan_force_max ] && echo yes || echo no)"
}
temps() {
  for z in /sys/class/thermal/thermal_zone*; do
    t=$(cat $z/type 2>/dev/null)
    case "$t" in CPU-therm|GPU-therm|AUX-therm|AO-therm|thermal_fan_est|thermal-fan-est)
      v=$(cat $z/temp 2>/dev/null); case "$v" in ''|*[!0-9]*) ;; *) printf '%s=%s ' "$t" "$v" ;; esac ;;
    esac
  done
  echo
}

case "$1" in
on)
    nvpmodel -m 8 || { echo "nvpmodel -m 8 failed"; exit 1; }
    jetson_clocks --store "$STATE" 2>/dev/null
    jetson_clocks || { echo "jetson_clocks failed"; exit 1; }
    touch /tmp/fan_force_max
    sleep 4
    ;;
off)
    if [ -f "$STATE" ]; then jetson_clocks --restore "$STATE" 2>/dev/null; fi
    nvpmodel -m 5 || { echo "nvpmodel -m 5 failed"; exit 1; }
    # JetPack 4.x only: nvpmodel resets the legacy fan mode, so set it through nvpmodel. Not available on 5.x.
    if [ -d /sys/devices/pwm-fan ]; then nvpmodel -d cool || echo "nvpmodel -d cool failed"; fi
    rm -f /tmp/fan_force_max
    ;;
status) ;;
*)
    echo "usage: $0 on|off|status"; exit 2 ;;
esac

echo "nvpmodel: $(nvpmodel -q 2>/dev/null | grep -E 'Mode' | tr '\n' ' ')"
fan_state
temps
