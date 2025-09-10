PIDS=$(pgrep -f "${PROCESS_KEYWORD}")

for pid in $PIDS;do
    kill "$pid"
done