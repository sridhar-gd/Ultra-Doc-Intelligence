#!/bin/sh
set -e

# Ignore any accidental platform command override and always boot API.
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips="*"
