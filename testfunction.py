def format_extracted_number(text: str) -> str:
    parts = text.split()
    valid_numbers = []
    dot_part = None
    for part in parts:
        number = ''.join(filter(str.isdigit, part))
        if len(number) == 8:
            if '.' in part:
                dot_part = part
                valid_numbers = [part]  # 重置 valid_numbers 只包含带点的部分
                break  # 找到带点的部分后立即退出循环
            elif not dot_part:
                valid_numbers.append(part)

    if len(valid_numbers) == 2 and dot_part:
        return dot_part
    return ' '.join([''.join(filter(str.isdigit, part)) for part in valid_numbers])



txt = '103., 765 570 PR 65241701 20240423 40550104 2417 45kg'
a = format_extracted_number(txt)
print("a", type(a), a, len(a))

