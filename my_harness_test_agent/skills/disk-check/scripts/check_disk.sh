#!/usr/bin/env bash
# Проверка свободного места на диске (корневой раздел /)
set -euo pipefail

df -h /
