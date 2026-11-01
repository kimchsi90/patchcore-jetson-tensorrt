#!/bin/sh
# Temperature-driven fan control for the Xavier NX dev kit (replaces nvfancontrol, which is not installed here).
# Fan is off when cool, steps up when the hottest CPU/GPU/AUX zone gets warm, with 4 C hysteresis when stepping down.
FAN=/sys/devices/pwm-fan
HW=$FAN/hwmon/hwmon2
LOG=/tmp/fan_daemon.log
echo 0 > $HW/temp_control || { echo "cannot switch to manual fan control" >> $LOG; exit 1; }
# thresholds in milli-C and the pwm for each level (index 0 = off). pwm values come from pwm_rpm_table (140/170/200/255).
T1=48000; T2=56000; T3=65000; T4=75000
P0=0; P1=140; P2=170; P3=200; P4=255
level=0
echo "$(date +%T) fan_daemon start" >> $LOG
while true; do
  # nvpmodel can switch the fan back to automatic mode, so assert manual control every cycle
  [ "$(cat $HW/temp_control)" = "0" ] || { echo 0 > $HW/temp_control; level=-1; }
  max=0
  for z in $(ls -d /sys/devices/virtual/thermal/thermal_zone*); do
    case "$(cat $z/type)" in CPU-therm|GPU-therm|AUX-therm)
      t=$(cat $z/temp); [ "$t" -gt "$max" ] && max=$t ;;
    esac
  done
  want=0
  [ "$max" -ge $T1 ] && want=1
  [ "$max" -ge $T2 ] && want=2
  [ "$max" -ge $T3 ] && want=3
  [ "$max" -ge $T4 ] && want=4
  if [ "$want" -lt "$level" ]; then
    # step down only when 4 C below the threshold of the current level
    case $level in 1) th=$T1;; 2) th=$T2;; 3) th=$T3;; 4) th=$T4;; esac
    if [ "$max" -gt $((th - 4000)) ]; then want=$level; fi
  fi
  # benchmark runs: perf_mode.sh on creates this flag to force the fan to maximum
  if [ -f /tmp/fan_force_max ]; then want=4; fi
  if [ "$want" != "$level" ]; then
    level=$want
    case $level in 0) p=$P0;; 1) p=$P1;; 2) p=$P2;; 3) p=$P3;; 4) p=$P4;; esac
    echo $p > $FAN/target_pwm
    echo "$(date +%T) temp=$max level=$level pwm=$p" >> $LOG
  fi
  sleep 3
done