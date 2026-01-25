#!/bin/bash

# 탐색 루트
SEARCH_ROOT="/DATA2/datasets/PR/STheReO"

echo "🔍 Searching for .tar.gz files..."
echo "==================================================="

find "$SEARCH_ROOT" -type f -name "*.tar.gz" | sort | while read FILE; do
  
  # 파일이 있는 디렉토리 경로
  DIR=$(dirname "$FILE")
  
  # 파일명만 추출 (예: dataset.tar.gz)
  BASENAME=$(basename "$FILE")
  
  # 파일명에서 .tar.gz 제거하여 예상 폴더명 추출 (예: dataset)
  FOLDER_NAME="${BASENAME%.tar.gz}"
  
  # 이미 압축 해제된 폴더가 있는지 확인
  if [ -d "$DIR/$FOLDER_NAME" ]; then
    echo "⏭️  Skipping: $BASENAME (Folder '$FOLDER_NAME' already exists)"
    continue
  fi
  
  echo "📂 Found: $FILE"
  echo "   Extracting to $DIR ..."
  
  # 압축 해제
  tar -xf "$FILE" -C "$DIR"

  if [ $? -eq 0 ]; then
    echo "✅ Extracted successfully!"
  else
    echo "❌ Failed to extract $FILE"
  fi
  
  echo "---------------------------------------------------"

done

echo "🎉 Done."