#!/bin/sh
# Usage (as root): perf_mode.sh on | off | status
# on     : max power mode + jetson_clocks + fan forced to max. Use only around benchmark runs.
# off    : back to MODE_10W_DESKTOP, default clocks, fan "cool" profile (fan runs only when the board warms up).
# status : print mode, fan state and temperatures.
# Mode IDs are for Xavier NX (nvpmodel.conf): 5 = MODE_10W_DESKTOP, 8 = MODE_20W_6CORE.
FAN=/sys/devices/pwm-fan
STATE=/tmp/jetson_clocks_state.conf

case "$1" in
on)
    nvpmodel -m 8 || { echo "nvpmodel -m 8 failed"; exit 1; }
    jetson_clocks --store "$STATE" 2>/dev/null
    jetson_clocks || { echo "jetson_clocks failed"; exit 1; }
    # fan_daemon.sh (temperature-driven) forces the fan to maximum while this flag exists
    touch /tmp/fan_force_max
    sleep 4
    ;;
off)
    if [ -f "$STATE" ]; then jetson_clocks --restore "$STATE" 2>/dev/null; fi
    nvpmodel -m 5 || { echo "nvpmodel -m 5 failed"; exit 1; }
    # nvpmodel resets the fan mode to quiet, so set it through nvpmodel (a direct sysfs write is overwritten).
    nvpmodel -d cool || { echo "nvpmodel -d cool failed"; exit 1; }
    # back to temperature-driven fan steps (fan_daemon.sh); the fan stops again when the board is cool
    rm -f /tmp/fan_force_max
    ;;
status) ;;
*)
    echo "usage: $0 on|off|status"; exit 2 ;;
esac

echo "nvpmodel: $(nvpmodel -q 2>/dev/null | head -2 | tr '\n' ' ')"
echo "fan: profile=$(cat $FAN/fan_profile) target_pwm=$(cat $FAN/target_pwm) cur_pwm=$(cat $FAN/cur_pwm)"
for z in /sys/devices/virtual/thermal/thermal_zone*; do
    t=$(cat $z/type)
    case "$t" in CPU-therm|GPU-therm|AUX-therm|AO-therm) printf '%s=%s ' "$t" "$(cat $z/temp)" ;; esac
done
echo
