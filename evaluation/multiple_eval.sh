#!/bin/bash

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
  echo "오류: 유효한 폴더 경로를 입력하세요."
  exit 1
fi

# .txt 파일만 대상으로 처리
find "$TARGET_DIR" -type f -name "*.txt" | while read file; do
  python get_results.py "$file"
  echo ""
done
