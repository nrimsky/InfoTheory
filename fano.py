def shannon_fano_encode(counts):
    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    l = sum([c[1] for c in counts])
    codes = [""] * len(counts)
    codes = _encode(codes, counts, l)
    codes = {counts[i][0]: codes[i] for i in range(len(counts))}
    return codes


def _encode(codes, counts, l):
    if len(codes) == 0:
        return codes
    if len(codes) == 1:
        return codes
    if len(codes) == 2:
        codes[0] += "0"
        codes[1] += "1"
        return codes
    tot = 0
    mid = 0
    for i, (_, c) in enumerate(counts):
        if i == len(counts) - 2:
            mid = i
            break
        if tot >= l / 2:
            mid = i
            break
        codes[i] += "0"
        tot += c
    for i in range(mid, len(codes)):
        codes[i] += "1"
    codes[0: mid] = _encode(codes[0: mid], counts[0: mid], tot)
    codes[mid:] = _encode(codes[mid:], counts[mid:], l - tot)
    return codes


if __name__ == '__main__':
    p = {
        "a": 0.2,
        "b": 0.19,
        "c": 0.17,
        "d": 0.15,
        "e": 0.14,
        "f": 0.06,
        "g": 0.05,
        "h": 0.04
    }
    print(shannon_fano_encode(list(p.items())))
    p = {
        "a": 0.385,
        "b": 0.179,
        "c": 0.154,
        "d": 0.154,
        "e": 0.128
    }
    print(shannon_fano_encode(list(p.items())))
