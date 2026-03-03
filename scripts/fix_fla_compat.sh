#!/bin/bash
# fix_fla_compat.sh - Patch Zoology fla wrappers for fla 0.4.1
# Removes head_first=False which fla 0.4.1 no longer accepts.
# Safe to run multiple times.
ZOOLOGY_DIR="${1:-/workspace/zoology}"
MIXERS="$ZOOLOGY_DIR/zoology/mixers"
echo "Patching fla wrappers..."
for f in gla.py delta_net.py gated_delta_net.py rwkv7.py; do
    filepath="$MIXERS/$f"
    [ ! -f "$filepath" ] && continue
    if grep -q head_first "$filepath"; then
        sed -i '/head_first=False/d' "$filepath"
        python3 -c "import re; t=open('$filepath').read(); t=re.sub(r',(\s*\))',r'\1',t); open('$filepath','w').write(t)"
        echo "  FIXED $f"
    else
        echo "  OK $f"
    fi
done
echo "Done."
