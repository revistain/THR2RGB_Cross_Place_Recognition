import matplotlib.pyplot as plt
import numpy as np

# 가로(행)와 세로(열) 단서
row_clues = [[3, 3], [2, 1, 4], [8, 4], [2, 1], [1, 5], [3, 3], [2, 4, 1], [6, 4, 2], [1, 2], [5, 1]]
col_clues = [[1, 1], [2, 1], [3, 1], [1, 1], [1, 1], [1, 3], [1, 2], [1, 1], [4, 1], [2, 2, 1], [4, 1], [2, 6], [7, 1], [3, 1, 1], [2, 1, 2], [1, 4]]

# 그리드 크기 설정
rows = len(row_clues)
cols = len(col_clues)

fig, ax = plt.subplots(figsize=(10, 6))

# 격자 그리기
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.grid(color='black', linestyle='-', linewidth=1)

# 가로 단서 텍스트 삽입 (왼쪽)
for i, clue in enumerate(row_clues):
    clue_text = " ".join(map(str, clue))
    ax.text(-0.8, i, clue_text, va='center', ha='right', fontsize=12, fontweight='bold')

# 세로 단서 텍스트 삽입 (위쪽)
for j, clue in enumerate(col_clues):
    clue_text = "\n".join(map(str, clue))
    ax.text(j, -0.8, clue_text, va='bottom', ha='center', fontsize=12, fontweight='bold')

# 축 설정 및 디자인 다듬기
ax.set_xlim(-0.5, cols - 0.5)
ax.set_ylim(rows - 0.5, -0.5) # 위에서 아래로 행 번호 증가
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)
plt.title("Cyber Labyrinth - Quiz 10 Nonogram", fontsize=16, pad=50)

# 시각화 창 띄우기
plt.savefig("test.png")