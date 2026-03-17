# UNet (ResNet34 Encoder) 아키텍처 블록 다이어그램 생성 스크립트
# - Encoder/Bottleneck/Decoder/Skip Connection 구조를 matplotlib으로 시각화
# - 결과: modelresult/unet_architecture.png
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# 색상
C_INPUT   = '#4CAF50'   # 초록
C_ENC     = '#2196F3'   # 파랑
C_BOTTLE  = '#FF5722'   # 빨강
C_DEC     = '#FF9800'   # 주황
C_OUTPUT  = '#9C27B0'   # 보라
C_SKIP    = '#78909C'   # 회색
C_TEXT    = 'white'

def draw_block(ax, x, y, w, h, label, sublabel, color, fontsize=14):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='black', linewidth=2.0, alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.15, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=C_TEXT)
    ax.text(x + w/2, y + h/2 - 0.25, sublabel,
            ha='center', va='center', fontsize=11, color=C_TEXT)

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_skip(ax, x1, y1, x2, y2):
    """skip connection (점선 화살표)"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_SKIP,
                                lw=2.5, linestyle='dashed'))

# 블록 크기
BW = 2.8   # 블록 너비
BH = 0.9   # 블록 높이

# ===== 왼쪽: Encoder =====
enc_x = 1.0
enc_blocks = [
    ("Input",        "3 × 1280 × 1280",  C_INPUT,  10.0),
    ("Encoder Block 1", "64 × 640 × 640",   C_ENC,    8.7),
    ("Encoder Block 2", "128 × 320 × 320",  C_ENC,    7.4),
    ("Encoder Block 3", "256 × 160 × 160",  C_ENC,    6.1),
    ("Encoder Block 4", "512 × 80 × 80",    C_ENC,    4.8),
]

for label, sublabel, color, y in enc_blocks:
    draw_block(ax, enc_x, y, BW, BH, label, sublabel, color)

# ===== 가운데 아래: Bottleneck =====
bot_x = 6.6
bot_y = 3.5
draw_block(ax, bot_x, bot_y, BW, BH, "Bottleneck", "512 × 40 × 40", C_BOTTLE, fontsize=15)

# ===== 오른쪽: Decoder =====
dec_x = 12.2
dec_blocks = [
    ("Decoder Block 4", "256 × 80 × 80",    C_DEC, 4.8),
    ("Decoder Block 3", "128 × 160 × 160",  C_DEC, 6.1),
    ("Decoder Block 2", "64 × 320 × 320",   C_DEC, 7.4),
    ("Decoder Block 1", "32 × 640 × 640",   C_DEC, 8.7),
]

for label, sublabel, color, y in dec_blocks:
    draw_block(ax, dec_x, y, BW, BH, label, sublabel, color)

# ===== Output =====
draw_block(ax, dec_x, 10.0, BW, BH, "Output (Sigmoid)", "1 × 1280 × 1280", C_OUTPUT)

# ===== Encoder 아래 화살표 (위→아래) =====
for i in range(len(enc_blocks) - 1):
    y1 = enc_blocks[i][3]
    y2 = enc_blocks[i+1][3] + BH
    draw_arrow(ax, enc_x + BW/2, y1, enc_x + BW/2, y2)

# Encoder → Bottleneck
draw_arrow(ax, enc_x + BW, enc_blocks[-1][3] + BH/2,
           bot_x, bot_y + BH/2)

# Bottleneck → Decoder
draw_arrow(ax, bot_x + BW, bot_y + BH/2,
           dec_x, dec_blocks[0][3] + BH/2)

# ===== Decoder 위 화살표 (아래→위) =====
for i in range(len(dec_blocks) - 1):
    y1 = dec_blocks[i][3] + BH
    y2 = dec_blocks[i+1][3]
    draw_arrow(ax, dec_x + BW/2, y1, dec_x + BW/2, y2)

# Decoder → Output
draw_arrow(ax, dec_x + BW/2, dec_blocks[-1][3] + BH,
           dec_x + BW/2, 10.0)

# ===== Skip Connections (Encoder → Decoder) =====
skip_pairs = [
    (enc_blocks[4][3], dec_blocks[0][3]),  # Block4 → Dec4
    (enc_blocks[3][3], dec_blocks[1][3]),  # Block3 → Dec3
    (enc_blocks[2][3], dec_blocks[2][3]),  # Block2 → Dec2
    (enc_blocks[1][3], dec_blocks[3][3]),  # Block1 → Dec1
]

for enc_y, dec_y in skip_pairs:
    draw_skip(ax, enc_x + BW, enc_y + BH/2,
              dec_x, dec_y + BH/2)

# ===== Skip Connection 라벨 =====
ax.text(8.0, 8.9, "Skip Connection\n(Concatenate)", ha='center', va='center',
        fontsize=13, color=C_SKIP, fontstyle='italic')

# ===== 타이틀 =====
ax.text(8.0, 0.8, "U-Net Architecture (ResNet34 Encoder)",
        ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(8.0, 0.2, "Total params: 24,436,369  |  Input: 3×1280×1280  |  Output: 1×1280×1280 (Binary Mask)",
        ha='center', va='center', fontsize=13, color='gray')

# ===== 범례 =====
legend_items = [
    mpatches.Patch(color=C_INPUT, label='Input'),
    mpatches.Patch(color=C_ENC, label='Encoder (ResNet34)'),
    mpatches.Patch(color=C_BOTTLE, label='Bottleneck'),
    mpatches.Patch(color=C_DEC, label='Decoder'),
    mpatches.Patch(color=C_OUTPUT, label='Output'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=13, framealpha=0.9)

# ===== 저장 =====
save_path = "/workspace/BEADtrain/modelresult/unet_architecture2.png"
# 호스트용 경로도
save_path_host = "/home/ho/BEADtrain/modelresult/unet_architecture.png"

for p in [save_path, save_path_host]:
    try:
        plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"저장: {p}")
    except Exception:
        pass

plt.close()
print("완료!")
