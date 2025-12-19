#!/bin/bash

# 총 16시간 = 57600초
hours=16
duration=$((hours * 60 * 60))
interval=60
end_time=$((SECONDS + duration))

# 로그 파일 지정
logfile="nvidia_smi_log_$(date +%Y%m%d_%H%M%S).log"

echo "Logging GPU stats every $interval seconds for $hours hours..."
while [ $SECONDS -lt $end_time ]; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$logfile"
    nvidia-smi >> "$logfile"
    echo "" >> "$logfile"
    sleep $interval
done

echo "Done. Log saved to $logfile"

