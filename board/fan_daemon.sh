#!/bin/sh
# Fan control for the Xavier NX dev kit (nvfancontrol is missing or broken on these images). Runs as root, ideally as a systemd service.
# Fan interface:
#   JetPack 5.x: hwmon device named "pwmfan" (pwm1, 0-255); rpm in hwmon "pwm_tach"
#   JetPack 4.x: /sys/devices/pwm-fan/target_pwm with temp_control=0
# Heat input:
#   1. real temperatures when CPU-therm / GPU-therm zones exist (4 C hysteresis)
#   2. otherwise board power from the ina3221 (VDD_IN) as a proxy, because on one JetPack 5.1.4 boot the CPU/GPU thermal zones were
#      missing and thermal_fan_est stayed at 25 C. Proxy levels have 1.5 W hysteresis and a 60 s minimum dwell for thermal inertia.
# Benchmarks create /tmp/fan_force_max (perf_mode.sh on) to force the maximum.
LOG=/tmp/fan_daemon.log
PWMF=""; INA=""
for d in /sys/class/hwmon/hwmon*; do
  case "$(cat $d/name 2>/dev/null)" in pwmfan) PWMF=$d/pwm1 ;; ina3221) INA=$d ;; esac
done
LEGACY=/sys/devices/pwm-fan
if [ -z "$PWMF" ] && [ -d "$LEGACY" ]; then
  HW=$(ls -d $LEGACY/hwmon/hwmon* 2>/dev/null | head -1)
  echo 0 > $HW/temp_control || { echo "cannot switch to manual fan control" >> $LOG; exit 1; }
  PWMF=$LEGACY/target_pwm
fi
[ -n "$PWMF" ] || { echo "no fan pwm interface found" >> $LOG; exit 1; }

# pwm per level (index 0 = off)
P0=0; P1=130; P2=170; P3=200; P4=255
# temperature thresholds, milli-C
T1=48000; T2=56000; T3=65000; T4=75000
# power thresholds, mW
W1=6500; W2=9000; W3=11500; W4=14500

level=-1; last_change=0
echo "$(date +%T) fan_daemon start, pwm file $PWMF" >> $LOG

read_temp() {  # prints the hottest valid CPU/GPU/AUX zone in milli-C, or 0 when none exists
  max=0
  for z in /sys/class/thermal/thermal_zone*; do
    case "$(cat $z/type 2>/dev/null)" in CPU-therm|GPU-therm|AUX-therm)
      t=$(cat $z/temp 2>/dev/null)
      case "$t" in ''|*[!0-9]*) ;; *) [ "$t" -gt "$max" ] && max=$t ;; esac ;;
    esac
  done
  echo $max
}
read_power() {  # VDD_IN in mW
  [ -n "$INA" ] || { echo 0; return; }
  v=$(cat $INA/in1_input 2>/dev/null); i=$(cat $INA/curr1_input 2>/dev/null)
  case "$v$i" in ''|*[!0-9]*) echo 0 ;; *) echo $(( v * i / 1000 )) ;; esac
}

while true; do
  # legacy kernels: nvpmodel can switch the fan back to automatic mode, so assert manual control every cycle
  if [ -n "$HW" ] && [ "$(cat $HW/temp_control 2>/dev/null)" != "0" ]; then echo 0 > $HW/temp_control; level=-1; fi
  temp=$(read_temp)
  now=$(date +%s)
  if [ "$temp" -gt 0 ]; then
    src=temp; val=$temp; a=$T1; b=$T2; c=$T3; d=$T4; hyst=4000
  else
    src=power; val=$(read_power); a=$W1; b=$W2; c=$W3; d=$W4; hyst=1500
  fi
  want=0
  [ "$val" -ge $a ] && want=1
  [ "$val" -ge $b ] && want=2
  [ "$val" -ge $c ] && want=3
  [ "$val" -ge $d ] && want=4
  if [ "$want" -lt "$level" ]; then
    case $level in 1) th=$a;; 2) th=$b;; 3) th=$c;; 4) th=$d;; *) th=0;; esac
    if [ "$val" -gt $((th - hyst)) ]; then want=$level; fi
    # power is only a proxy: do not step down before 60 s have passed since the last change
    if [ "$src" = power ] && [ $((now - last_change)) -lt 60 ]; then want=$level; fi
  fi
  if [ -f /tmp/fan_force_max ]; then want=4; fi
  if [ "$want" != "$level" ]; then
    level=$want; last_change=$now
    case $level in 0) p=$P0;; 1) p=$P1;; 2) p=$P2;; 3) p=$P3;; 4) p=$P4;; esac
    echo $p > $PWMF
    echo "$(date +%T) src=$src value=$val level=$level pwm=$p" >> $LOG
  fi
  sleep 3
done
