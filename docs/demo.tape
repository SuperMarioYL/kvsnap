Output assets/demo.gif
Set FontSize 15
Set Width 1000
Set Height 500
Set Theme Dracula
Set Padding 20

# --- KVSnap happy path (offline --mock: no GPU/vLLM needed) ---
Type "kvsnap --version"
Enter
Sleep 500ms

Type "# 1. checkpoint a (mock) GLM-5.3 KV cache to disk"
Enter
Type "kvsnap save --mock --layers 3 --blocks 8 --dir /tmp/kvsnap-demo"
Enter
Sleep 800ms

Type "# 2. list saved sessions"
Enter
Type "kvsnap list --dir /tmp/kvsnap-demo"
Enter
Sleep 600ms

Type "# 3. inspect the manifest (arch fingerprint + blobs)"
Enter
Type "kvsnap info --dir /tmp/kvsnap-demo"
Enter
Sleep 700ms

Type "# 4. integrity-check every blob (xxhash64)"
Enter
Type "kvsnap verify --dir /tmp/kvsnap-demo"
Enter
Sleep 700ms

Type "# 5. restore into a fresh backend (round-trip, no re-prefill)"
Enter
Type "kvsnap restore --mock --dir /tmp/kvsnap-demo"
Enter
Sleep 900ms

Type "# production: kill -USR1 <pid> saves  ·  restart: kvsnap serve --model glm-5.3-1m --restore default"
Enter
Sleep 2000ms
