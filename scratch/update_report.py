with open('kl8_report.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """    ordered_nums = []
    for g in groups:
        for m in g["members"]:
            if m not in ordered_nums:
                ordered_nums.append(m)
    L.append(f"  主攻去重精选码 ({len(ordered_nums)}个): " + " ".join("%02d" % n for n in ordered_nums))
    L.append("  实战使用建议: 优先重注 [T1核心] 连号组合，搭配 [T2防守] 覆盖其它宏观区间，实现防空门对冲。")
    L.append("")"""

replacement = """    ordered_nums = []
    for g in groups:
        for m in g["members"]:
            if m not in ordered_nums:
                ordered_nums.append(m)
    L.append(f"  主攻去重精选码 ({len(ordered_nums)}个): " + " ".join("%02d" % n for n in ordered_nums))
    L.append("  实战使用建议: 优先重注 [T1核心] 连号组合，搭配 [T2防守] 覆盖其它宏观区间，实现防空门对冲。")
    L.append("")
    L.append("  ★ 集群三元组·纯净可复制清单 (格式四 · Top 8):")
    for t in ti["triples"][:8]:
        L.append(f"    {'-'.join('%02d'%m for m in t['members'])}")
    L.append("")"""

# Normalize CRLF
text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')
replacement_norm = replacement.replace('\r\n', '\n')

if target_norm in text_norm:
    text_norm = text_norm.replace(target_norm, replacement_norm, 1)
    with open('kl8_report.py', 'w', encoding='utf-8', newline='\n') as f:
        f.write(text_norm)
    print('Updated kl8_report.py successfully')
else:
    print('Target not found')
