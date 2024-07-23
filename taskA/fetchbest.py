best = {}

with open('ans.txt', 'r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        if line[0] == '*' or line[0] == '=':
            continue
        if line[:3] == 'avg':
            lang = 'avg'
            if lang in best.keys():
                best[lang] = score if score > best[lang] else best[lang]
            else:
                best.setdefault(lang, score)
        lang = line[:3]
        score = line[line.find('=') + 1 : line.find(',')]
        if lang in best.keys():
            best[lang] = score if score > best[lang] else best[lang]
        else:
            best.setdefault(lang, score)
print(best)