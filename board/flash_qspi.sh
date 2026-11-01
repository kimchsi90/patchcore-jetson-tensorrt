#!/bin/sh
# Writes the Xavier NX QSPI image, retries up to 3 times, verifies by reading back all 32 MiB.
D=/home/<user>/qspi
IMG=$D/jetson-xavier-nx-devkit.spi.img
WANT=ad292462122cc06e85d5647e24c4708a47578568545fc1536e0be615bcdf7c25
echo "start $(date)"
GOT0=$(sha256sum "$IMG" | cut -d' ' -f1)
if [ "$GOT0" != "$WANT" ]; then echo "image hash mismatch: $GOT0"; echo RESULT=ABORTED; exit 1; fi
n=0
while [ "$n" -lt 3 ]; do
  n=$((n+1))
  echo "attempt $n $(date)"
  if flash_eraseall /dev/mtd0 && flashcp -v "$IMG" /dev/mtd0; then
    GOT=$(dd if=/dev/mtd0 bs=65536 count=512 2>/dev/null | sha256sum | cut -d' ' -f1)
    echo "readback $GOT"
    if [ "$GOT" = "$WANT" ]; then echo "end $(date)"; echo RESULT=SUCCESS; exit 0; fi
  fi
  echo "attempt $n failed"
done
echo RESULT=FAILED
exit 1